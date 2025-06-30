"""-----------------------------------------------------------------------------
Dictionary generation class for a MR fingerprint using a bloch simiulation of a 
two compartment model with a semipermeable barrier 

VARIATIONS IN: 
    - vb: percentage blood volume 
    - tb: intravascular water residence time 
    - T1t: T1 of tissue compartment
    - T1b: T1 of blood compartment 
    - B1+: B1 multiplication factor

Author: Ela Kanani, adapted from Emma Thomson
Year: 2025
Institution: Centre for Medical Image Computing: University College London
Email: e.kanani.21@ucl.ac.uk
----------------------------------------------------------------------------"""

''' -----------------------------PACKAGES--------------------------------- '''

""" PACKAGE IMPORTS """
import numpy as np
import sys
import os
import platform
import warnings
from scipy import signal, io
import numba_helper as nbh
from line_profiler import LineProfiler


class DictionaryGeneratorFast():
    # Initialise the class
    def __init__(self, t1Array, t2Array, t2StarArray, noOfIsochromatsX,
                noOfIsochromatsY, noOfIsochromatsZ, noOfRepetitions, noise, perc, res,
                multi, inv, CSFnullswitch, sliceProfileSwitch, samples, dictionaryId, sequence, instance, readTRFA= True, trArray = None, faArray = None):
        """
        Parameters:
        -----------
        t1Array : numpy nd array, shape (2,)
            Array of T1 values for the tissue and blood compartments
        t2Array : numpy nd array, shape (2,)
            Array of T2 values for the tissue and blood compartments
        t2StarArray : numpy nd array, shape (2,)
            Array of T2* values for the tissue and blood compartments
        noOfIsochromatsX : int
            Number of isochromats in the x direction
        noOfIsochromatsY : int
            Number of isochromats in the y direction
        noOfIsochromatsZ : int
            Number of isochromats in the z direction
        noOfRepetitions : int
            TR train length
        noise : int
            Number of noise levels (set to one for dictionary generation)
        perc : int
            Percentage blood volume % (NOTE: it is divided by ten)
        res : int
            intravascular water residence time UNIT: ms
        multi : float
            Multiplication factor for the B1 value
        inv : bool
            Inversion pulse switch
        CSFnullswitch : bool
            CSF nulling switch
        sliceProfileSwitch : bool
            Slice profile switch
        samples : int
            Number of noise samples generated (set to one for dictionary generation)
        dictionaryId : str
            Dictionary ID
        sequence : str
            MRF sequence to use
        instance : int
            Instance number for multiple instances of the same code to run
        readTRFA : bool
            Whether to read in the TR and FA values from a file or as an input array
        trArray : numpy nd array, shape (noOfRepetitons,)
            The TR variation used in the protocol (equal to length to the number of dynamics). Optional input, else it is read from file.
        faArray : numpy nd array, shape (noOfRepetitons,)
            The FA variation used in the protocol (equal to length to the number of dynamics). Optional input, else it is read from file.
        """    
        
        # Set the input arguments as class variables
        self.t1Array = t1Array
        self.t2Array = t2Array
        self.t2StarArray = t2StarArray
        self.noOfIsochromatsX = noOfIsochromatsX
        self.noOfIsochromatsY = noOfIsochromatsY
        self.noOfIsochromatsZ = noOfIsochromatsZ
        self.noOfRepetitions = noOfRepetitions
        self.noise = noise
        self.perc = perc
        self.res = res
        self.multi = multi
        self.inv = inv
        self.CSF_null = CSFnullswitch
        self.samples = samples
        self.dictionaryId = dictionaryId
        self.sequence = sequence
        self.instance = instance
        self.sliceProfileSwitch = sliceProfileSwitch

        # What type of file saving structure. Npy is the original method. hdf5 is the newer more efficient method. 'None' means do not save
        self.filesavetype = 'hdf5'
        
    
        # PARAMETER DECLARATION
        
        ### set the rf pulse duration
        ### TO DO: need to allow for variable slice profile and pulse duration
        self.pulseDuration = 0 #UNIT ms 
        
        ### calculated position array used for the prcession according to spatial gradients     
        [positionArrayXHold ,positionArrayYHold] = \
                    np.meshgrid(range(noOfIsochromatsX),range(noOfIsochromatsY))
        self.positionArrayX = positionArrayXHold - ((noOfIsochromatsX/2))
        self.positionArrayY = positionArrayYHold - ((noOfIsochromatsY/2))

        
        ### Time increment
        self.deltaT = 1 #1ms dt

        #Initially gradient is 0 (while pulse is on)
        self.gradientX = 0 #T/m
        self.gradientY = 0 #T/m
        

        ### Set echo time (must be divisible by deltaT) 
        ## TO DO: Need to remove hard coding for TE=2 out of calculation code 
        self.TE = 2 #ms   

        # Flag for returning complex or magnitude fingerprint
        self.complexFing = False

        

        """---------------------------OPEN ARRAYS-----------------------------"""
        ### This is defined as a unit vector along the z-axis
        vecM = np.float32([[0],[0],[1]])
        
        ## Define arrays for blood and tissue 
        self.vecMArrayBlood = np.tile(vecM.T, [int(perc*noOfIsochromatsX/100), noOfIsochromatsY, noOfIsochromatsZ, 1])
        self.vecMArrayTissue = np.tile(vecM.T, [int(noOfIsochromatsX-perc*noOfIsochromatsX/100), noOfIsochromatsY, noOfIsochromatsZ, 1] )
            

        ### Expand the dimensions for multiplication 
        self.vecMArrayTissue = np.expand_dims(self.vecMArrayTissue, axis=4)
        self.vecMArrayBlood = np.expand_dims(self.vecMArrayBlood, axis=4)

        ### FA array

        if readTRFA == True:
            # if the flip angle and repetition time trains are to be read from a file
            faString = './functions/holdArrays/faArray_' + str(instance) + '.npy'
            #faString = './functions/holdArrays/faArray_' + str(instance) + '.npy'
            self.faArray = np.load(faString) 

            ### Open and round TR array 
            trString = './functions/holdArrays/trArray_' + str(instance) + '.npy'
            trArray = np.load(trString)
            # Rounding is required in order to assure that TR is divisable by deltaT
            self.trRound = np.round(trArray, 0)

        elif readTRFA == False:
            # if the input TR and FA arrays are used instead of files 
            self.faArray = faArray
            self.trRound = np.round(trArray, 0)
            
        ### Open noise sample array
        self.noiseArray = np.load('./functions/holdArrays/noiseSamples.npy')

        # add flag for integer FA here
        
        ### Empty signal array to store all magnitization at all time points 
        self.signal = np.zeros([noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3, noOfRepetitions])

        '''
        SLICE PROFILE ARRAY READ IN
        '''

        

        if self.sliceProfileSwitch == 1: 
            # slice profile switch: i.e. if SPGRE or FISP slice profile is used, set the type
            if self.sequence == 'FISP':
                sliceProfilePath = '../sliceProfile/sliceProfileRFFISP.mat' #'../sliceProfile/sliceProfile.mat'
                self.sliceProfileType = 'FISP' # global variable to be used later on
                sliceProfileArray = io.loadmat(sliceProfilePath)['sliceProfile']
                self.sliceProfile = self.sampleSliceProfile(sliceProfileArray, sample = 'half') # sample across the isochromats

            elif self.sequence == 'SPGRE':
                self.sliceProfileType = 'SPGRE' 
                sliceProfilePath = '../sliceProfile/sliceProfileRFSPGRE.mat'
                sliceProfileArray = io.loadmat(sliceProfilePath)['sliceProfile']
                self.sliceProfile = self.sampleSliceProfile(sliceProfileArray, sample = 'half') # sample across the isochromats

                # Emma's implementation:
                #sliceProfileArray = io.loadmat(sliceProfilePath)['sliceProfile']
                #self.sliceProfile = self.sampleSliceProfile(sliceProfileArray, sample = 'half') # sample across the isochromats


        elif self.sliceProfileSwitch == 0: 
            #If you want flat slice profile then this (i.e. homogenous excitation profile across the slice thickness)
            self.sliceProfile = np.tile(np.expand_dims(np.round(np.linspace(1,90,90*100),0), axis=1),noOfIsochromatsZ)   
            self.sliceProfileType = 'None' # arbitrary so the code continues to work

        '''
        PEAK LOCATIONS
        '''
        
        ### Calculate the echo peak position peak position
        self.signalDivide = np.zeros([noOfRepetitions])
        for r in range(noOfRepetitions):  
            if self.inv == 0:
                self.signalDivide[r] = (sum(self.trRound[:r])+self.TE+self.pulseDuration)/self.deltaT
            else: 
                self.signalDivide[r] = (sum(self.trRound[:r])+self.TE+self.pulseDuration)/self.deltaT 

        
        
        return None 

    def sampleSliceProfile(self, fullsliceProfile, sample = 'half'):
        """
        Samples half of an symmetrical slice profile according to the number of samples
        in the z-direction.

        Parameters:
        -----------
        fullsliceProfile : numpy nd array, shape (9000, N)
            Slice profile for angles of 0.1:0.1:90 degrees with N samples
        
        sample : str
            What type of sample of the slice profile  ('half' or 'full')
           

        Returns:
        --------
        sliceProfile : numpy nd array, shape (9000, noOfIsochromatsZ)
            Sampled slice profile
            
        """

        if sample == 'half': # sample half of the profile
            endPoint = np.size(fullsliceProfile, 1)-1
            startPoint = (np.size(fullsliceProfile, 1)/2) #stepSize/2 
            profileSamples = np.linspace(startPoint, endPoint, self.noOfIsochromatsZ, endpoint=True, dtype=int)
            
        
        elif sample == "full": # sample the whole slice profile
            endPoint = np.size(fullsliceProfile, 1) - 1
            startPoint = 0
            profileSamples = np.linspace(startPoint, endPoint, self.noOfIsochromatsZ, dtype=int) 
        
        
        sampledSliceProfile = fullsliceProfile[:,profileSamples]
        #ßnp.savetxt('sliceProfileUsed.csv', sampledSliceProfile, delimiter=',')

        return sampledSliceProfile
    
    def invpulse(self, loop):
        """
        
        Application of an inversion pulse to a Bloch equation simulation with two compartments

        Parameters:
        -----------
        vecMArrayTissue : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3)
            Array of magnetization vectors for the tissue compartment
        vecMArrayBlood : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3)
            Array of magnetization vectors for the blood compartment
        loop : int
            Loop number in number of repetitions (TR)
        noOfIsochromatsZ : int
            Number of isochromats in the z direction
        multi : float
            Multiplier for the inversion pulse

        Returns:
        --------
        vecMArrayTissue : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3)
            Array of magnetization vectors for the tissue compartment
        vecMArrayBlood : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3)
            Array of magnetization vectors for the blood compartment
        """
        
        #180 pulse for inversion
        #because the 180 pulse is rubbish multiply value by 0.7
        #this is extremely crude - need to add either another parameter 
        # or manually code the IR pulse in the sequence code
        #thetaX = np.pi*self.multi*0.8*np.ones([self.noOfIsochromatsZ])
        
        if self.sequence == 'FISP':
            thetaX = np.pi*self.multi
            vecMRotation = np.array([[1, 0, 0], [0, np.cos(thetaX), np.sin(thetaX)], \
                                    [0, -np.sin(thetaX), np.cos(thetaX)]])
            vecMRotation = np.tile(vecMRotation,(self.noOfIsochromatsZ,1,1))

            # Updating the magnetization vector matricies
            #For tissue
            #self.vecMArrayTissue = np.matmul(vecMRotation,self.vecMArrayTissue)
            self.vecMArrayTissue = np.einsum("...ij,...j", vecMRotation, self.vecMArrayTissue[..., 0])[..., None]
            #For blood 
            #self.vecMArrayBlood = np.matmul(vecMRotation,self.vecMArrayBlood)
            self.vecMArrayBlood = np.einsum("...ij,...j", vecMRotation, self.vecMArrayBlood[..., 0])[..., None]

        if self.sequence == 'SPGRE':
            thetaX = np.pi*self.multi*0.8

            vecMRotation = np.array([[1, 0, 0], [0, np.cos(thetaX), np.sin(thetaX)], \
                                    [0, -np.sin(thetaX), np.cos(thetaX)]])
            vecMRotation = np.tile(vecMRotation,(self.noOfIsochromatsZ,1,1))

            # Updating the magnetization vector matricies
            #For tissue
            #self.vecMArrayTissue = np.matmul(vecMRotation,self.vecMArrayTissue)
            self.vecMArrayTissue = np.einsum("...ij,...j", vecMRotation, self.vecMArrayTissue[..., 0])[..., None]
            #For blood 
            #self.vecMArrayBlood = np.matmul(vecMRotation,self.vecMArrayBlood)
            self.vecMArrayBlood = np.einsum("...ij,...j", vecMRotation, self.vecMArrayBlood[..., 0])[..., None]

        return None
    
    def longTR(self, remainingDuration, totalTime, signal_flag = 'False'):
            """
            Calculation of the relaxation that occurs as the remaining TR plays out for a
            group of isochromats in a two compartment model.
            
            Parameters:
            -----------
            remainingDuration : float
                Remaining duration of the TR
            deltaT : int
                    Time increment
            gradientX : float
                    Gradient in the x direction
            gradientY : float
                    Gradient in the y direction
            positionArrayX : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY)
                    Array of x positions of isochromats
            positionArrayY : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY)
                    Array of y positions of isochromats
            vecMArrayTissue : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3)
                    Array of magnetization vectors for the tissue compartment
            vecMArrayBlood : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3)
                    Array of magnetization vectors for the blood compartment
            t1Array : numpy nd array, shape (2,)
                    Array of T1 values for the tissue and blood compartments
            t2StarArray : numpy nd array, shape (2,)
                    T2* values for the tissue and blood compartments
            totalTime : int
                    Total time passed

            Returns:
            --------
            vecMArrayTissue : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3)
                    Array of magnetization vectors for the tissue compartment
            vecMArrayBlood : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3)
                    Array of magnetization vectors for the blood compartment
            totalTime : int
                    Total time passed

            """

            #Update time passed 
            totalTime = totalTime + remainingDuration

            #For the tissue compartment
            #Set the relavent T1 and T2*
            t1 = self.t1Array[0]
            t2Star = self.t2StarArray[0]

            #Set a hold array
            vecMIsochromat = self.vecMArrayTissue

            # The magnitude change due to relaxation is then applied to each
            # coordinate
            vecMIsochromat[:,:,:,0,:] = np.exp(-(remainingDuration)/t2Star)*vecMIsochromat[:,:,:,0,:]
            vecMIsochromat[:,:,:,1,:] = np.exp(-(remainingDuration)/t2Star)*vecMIsochromat[:,:,:,1,:]
            vecMIsochromat[:,:,:,2,:] = (1-np.exp(-remainingDuration/t1))*1 + vecMIsochromat[:,:,:,2,:]*(np.exp(-remainingDuration/t1))
            #The stored array is then updated
            self.vecMArrayTissue = vecMIsochromat

            #For the blood compartment
            #Set the relavent T1 and T2*
            t1 = self.t1Array[1]
            t2Star = self.t2StarArray[1]

            #Set a hold array
            vecMIsochromat = self.vecMArrayBlood

            # The magnitude change due to relaxation is then applied to each
            # coordinate
            vecMIsochromat[:,:,:,0,:] = np.exp(-(remainingDuration)/t2Star)*vecMIsochromat[:,:,:,0,:]
            vecMIsochromat[:,:,:,1,:] = np.exp(-(remainingDuration)/t2Star)*vecMIsochromat[:,:,:,1,:]
            vecMIsochromat[:,:,:,2,:] = (1-np.exp(-remainingDuration/t1))*1  + vecMIsochromat[:,:,:,2,:]*(np.exp(-remainingDuration/t1))

            #The stored array is then updated
            self.vecMArrayBlood = vecMIsochromat

            #If the total time that has passed corresponds to the time at which
            # there is an echo peak:
            if int(totalTime/self.deltaT) in self.signalDivide and signal_flag == True: 
                vecMArray = np.concatenate((self.vecMArrayTissue,self.vecMArrayBlood),axis=0)
                #Get the index of the peak (what number peak is it?)
                signalDivide = list(self.signalDivide)
                ind = signalDivide.index(int(totalTime/self.deltaT))
                #Then input the magentization array at that time into the siganl
                # holder array
                try:
                    self.signal[:,0,:,:,ind] = np.squeeze(vecMArray)
                except:
                    self.signal[:,0,:,:,ind] =  np.expand_dims(np.squeeze(vecMArray), axis=1)
                
                #print(f"totalTime: {totalTime}, signalDivide: {self.signalDivide[ind]}")


            return totalTime    
    
    def rfpulse(self, loop):
        """
        Application of an RF pulse to a Bloch equation simulation with two compartments.

        Parameters:
        -----------
        vecMArrayTissue : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3)
            Array of magnetization vectors for the tissue compartment
        vecMArrayBlood : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3)
            Array of magnetization vectors for the blood compartment
        loop : int
            Loop number in number of repetitions
        faArray : numpy nd array, shape (noOfRepetitions,)
            Array of flip angles
        noOfIsochromatsZ : int
            Number of isochromats in the z direction
        sliceProfile : numpy nd array, shape (9000, noOfIsochromatsZ)
            Slice profile array
        multi : float
            Multiplication factor for the B1 value

        Returns:
        --------
        vecMArrayTissue : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3)
            Array of magnetization vectors for the tissue compartment
        vecMArrayBlood : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3)
            Array of magnetization vectors for the blood compartment
        
        """


        integerFlip = True # flag for slice profile of original FISP paper select false
    
        
        # Flag for integer flip angles
        if integerFlip == True and (self.sliceProfileType == 'FISP' or self.sliceProfileType == 'SPGRE'): # for my slice profile correction
            faInt = int(self.faArray[loop]*100)
            
            #Extract the flip angle of this loop (degrees)
            if faInt != 0:
                try: 
                    fa = self.multi*self.sliceProfile[faInt-1,:]

                except: 
                    fa = self.multi*np.ones([self.noOfIsochromatsZ])*180 
                    
            else: 
                fa = self.multi*np.zeros([self.noOfIsochromatsZ])*180

        elif self.sliceProfileSwitch == 0 and integerFlip == False:
            fa = self.multi*np.ones([self.noOfIsochromatsZ])*(self.faArray[loop])


        elif self.sliceProfileSwitch == 0 and integerFlip == True:
            fa = self.multi*np.ones(self.noOfIsochromatsZ)*(self.faArray[loop])
            fa = fa.astype(int)
            
        
        #Convert to radians
        thetaX = np.deg2rad(fa) #((fa/360)*2*np.pi)
     
        
        rotX = np.zeros([len(thetaX),3,3])
        sin_thetaX,  cos_thetaX = nbh.sincos(thetaX)
        rotX[:,0, 0] = 1
        rotX[:, 1, 1] = cos_thetaX
        rotX[:, 1, 2] = sin_thetaX
        rotX[:, 2, 1] = -sin_thetaX
        rotX[:, 2, 2] = cos_thetaX    

        # Updating the magnetization vector matrices
        #For tissue
        #self.vecMArrayTissue = np.matmul(rotX, self.vecMArrayTissue)
        self.vecMArrayTissue = np.einsum("...ij,...j", rotX, self.vecMArrayTissue[..., 0])[..., None]

        #For blood 
        #self.vecMArrayBlood = np.matmul(rotX, self.vecMArrayBlood)
        self.vecMArrayBlood = np.einsum("...ij,...j", rotX, self.vecMArrayBlood[..., 0])[..., None]

        return None
    
    def rf_spoil(self, loop):
        """
        Application of an RF spoiling to a Bloch equation simulation with two compartments

        Parameters:
        -----------
        vecMArrayTissue : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3)
            Array of magnetization vectors for the tissue compartment
        vecMArrayBlood : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3)
            Array of magnetization vectors for the blood compartment
        loop : int
            Loop number in number of repetitions

        Returns:
        --------
        vecMArrayTissue : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3)
            Array of magnetization vectors for the tissue compartment
        vecMArrayBlood : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3)
            Array of magnetization vectors for the blood compartment

        """
        #Using the rf phase formula developed by Zur et al (1991)
        # calculate the phase change for this particular repetition
        alpha0 = (123/360)*2*np.pi
        thetaZ = 0.5*alpha0*(loop**2+loop+2)
        
        """
        #Rotation matrices for this rotation
        rotX = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
        rotY = np.array([[np.cos(thetaZ), -np.sin(thetaZ), 0],\
                            [np.sin(thetaZ), np.cos(thetaZ), 0],\
                            [0, 0, 1]])
        #Combined rotation (in this case same as rotY)
        vecMIsochromatHold = np.matmul(rotY,rotX)
        # Updating the matrix so each time only the incremental rotation is
        # calculated. 
        vecMIsochromatHold = np.matmul(rotY,rotX) REDUNDANT BECAUSE ROT X IS IDENTITY MATRIX

        """
        rotY = np.array([[np.cos(thetaZ), -np.sin(thetaZ), 0],\
                            [np.sin(thetaZ), np.cos(thetaZ), 0],\
                            [0, 0, 1]])
    
            
        # Updating the magnetization vector matricies
        #For tissue
        #self.vecMArrayTissue = np.matmul(vecMIsochromatHold, self.vecMArrayTissue)
        self.vecMArrayTissue = np.einsum("...ij,...j", rotY, self.vecMArrayTissue[..., 0])[..., None]
        #For blood 
        #self.vecMArrayBlood = np.matmul(vecMIsochromatHold,self.vecMArrayBlood)
        self.vecMArrayBlood = np.einsum("...ij,...j", rotY, self.vecMArrayBlood[..., 0])[..., None]

        
        return None
    
    def rotation_calculations(self):
        """
        Calculation of the rotation matrices for each isochromat in a two compartment 
        model when gradients are applied leaving to spatially different gradient field 
        strengths.

        Parameters:
        -----------
        positionArrayX : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY)
            Array of x positions of isochromats
        positionArrayY : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY)
            Array of y positions of isochromats
        gradientX : float
            Gradient applied in the x direction
        gradientY : float
            Gradient applied in the y direction
        noOfIsochromatsZ : int
            Number of isochromats in the z direction
        deltaT : int
            Time increment
        
        Returns:
        --------
        precession : numpy nd array, shape (noOfIsochromatsY, noOfIsochromatsX, noOfIsochromatsZ, [3, 3])
            Array of rotation matrices [3 x 3] for each isochromat
        
        """
        
        #Find gradient field strength from both x and y gradients at each isochromat 
        # position
        gradientMatrix = self.gradientX*self.positionArrayX 
        gradientMatrix += self.gradientY*self.positionArrayY

         # Gyromagnetic ratio for proton 42.6 MHz/T 42.576
        omegaArray = np.repeat(np.expand_dims((42.6)*gradientMatrix, axis=2), self.noOfIsochromatsZ, axis=2) #* self.deltaT

        #for the precessions generate an array storing the 3x3 rotation matrix 
        #for each isochromat
        precession = np.zeros([np.size(self.positionArrayX,0), np.size(self.positionArrayY,1),self.noOfIsochromatsZ, 3,3])
        precession[:,:,:,2,2] = 1
   
        # compute the trigonometric functions for the rotation matrices
        sin_omega_deltaT, cos_omega_deltaT = nbh.sincos(omegaArray*self.deltaT)
  
        precession[:,:,:,0,0] = cos_omega_deltaT
        precession[:,:,:,0,1] = -sin_omega_deltaT
        precession[:,:,:,1,0] = sin_omega_deltaT
        precession[:,:,:,1,1] = cos_omega_deltaT
   
        return precession
    
    def applied_precession(self, gradientDuration, totalTime, signal_flag = True, saveMxMy =False, mxy_history_tissue=None, mxy_history_blood=None, current_tr_index=0):
        """
        Calculation of the precession of isochormats during the application of gradients
        for a two compartment model. Relaxation is considered.

        Parameters:
        -----------
        gradientDuration : float
            Duration of the gradient
        deltaT : int
            Time increment
        gradientX : float
            Gradient applied in the x direction
        gradientY : float
            Gradient applied in the y direction
        positionArrayX : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY)
            Array of x positions of isochromats
        positionArrayY : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY)
            Array of y positions of isochromats
        noOfIsochromatsZ : int
            Number of isochromats in the z direction
        vecMArrayTissue : numpy nd array, shape (int(noOfIsochromatsX-perc*noOfIsochromatsX/100), noOfIsochromatsY, noOfIsochromatsZ, 3, 1)
            Array of magnetization vectors for the tissue compartment
        vecMArrayBlood : numpy nd array, shape (int(perc*noOfIsochromatsX/100), noOfIsochromatsY, noOfIsochromatsZ, 3, 1)
            Array of magnetization vectors for the blood compartment
        t1Array : numpy nd array, shape (2,)
            Array of T1 values for the tissue and blood compartments
        t2StarArray : numpy nd array, shape (2,)
            Array of T2* values for the tissue and blood compartments
        signal : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3, noOfRepetitions)
            Array of signal to store all magnetisation at all time points 
        totalTime : int
            Total time passed
        signalDivide : numpy array, shape (noOfRepetitions,)
            Array where the echo peaks occur (echo peak position)
        signal_flag : bool
            Flag to indicate whether to store the signal or not
        
        Returns:
        --------
        vecMArrayTissue : numpy nd array, shape (noOfIsochromatsX in tissue, noOfIsochromatsY, noOfIsochromatsZ, 3)
            Array of magnetization vectors for the tissue compartment
        vecMArrayBlood : numpy nd array, shape (noOfIsochromatsX in blood, noOfIsochromatsY, noOfIsochromatsZ, 3)
            Array of magnetization vectors for the blood compartment
        signal : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3, noOfRepetitions)
            Array of signal to store all magnetisation at all time points 
        totalTime : int
            Total time passed
    
        """

        #Calculate the rotation matrices with different spatial matrices for each isochromat
        #in the array

        precession = self.rotation_calculations()
        
        """
        lp = LineProfiler()
        lp.add_function(self.rotation_calculations)
        lp_wrapper = lp(self.rotation_calculations)
        precession = lp_wrapper()
        lp.dump_stats("profile_results.txt")
        lp.print_stats()
        """
        
        #Transpose to correct shape
        precession = precession.transpose(1,0,2,4,3)
        
        #Separate the large precession array into the blood and tissue compartments
        precessionBlood = precession[:np.size(self.vecMArrayBlood,0),:, :, :]
        precessionTissue = precession[np.size(self.vecMArrayBlood,0):,:, :, :]


        # Pre calculate the exponential multipliers
        #For the tissue compartment
        #Set the relavent T1 and T2*
        t1 = self.t1Array[0]
        t2Star = self.t2StarArray[0]
        exp_delta_t_t2_star_tissue = np.exp((-self.deltaT)/t2Star)
        exp_delta_t_t1_tissue = (np.exp(-self.deltaT/t1))
        one_minus_exp_delta_t_t1_tissue = (1-np.exp(-self.deltaT/t1))

        #For the blood compartment
        #Set the relavent T1 and T2*
        t1 = self.t1Array[1]
        t2Star = self.t2StarArray[1]
        exp_delta_t_t2_star_blood = np.exp((-self.deltaT)/t2Star)
        exp_delta_t_t1_blood = (np.exp(-self.deltaT/t1))
        one_minus_exp_delta_t_t1_blood = (1-np.exp(-self.deltaT/t1))
        
        tStep_i = 0 # index to save Mxy for debugging

        #For each time step
        for tStep in range(int(gradientDuration/self.deltaT)):
            
                #Update time passed
                totalTime = totalTime + self.deltaT

                if saveMxMy == True:
                    # Calculate Mxy magnitude (example: average over isochromats - adjust as needed)
                    mxy_mag_tissue = np.mean(np.sqrt(self.vecMArrayTissue[:,:,:,0,:]**2 + self.vecMArrayTissue[:,:,:,1,:]**2))
                    mxy_mag_blood = np.mean(np.sqrt(self.vecMArrayBlood[:,:,:,0,:]**2 + self.vecMArrayBlood[:,:,:,1,:]**2))
                    # Store in history arrays
                    mxy_history_tissue[current_tr_index, tStep_i] = mxy_mag_tissue
                    mxy_history_blood[current_tr_index, tStep_i] = mxy_mag_blood
                    tStep_i += 1 # Increment time step index
                
                #Multiply by the precession rotation matrix (incremental for each deltaT)
                #vecMIsochromat = np.matmul(precessionTissue, self.vecMArrayTissue)
                vecMIsochromat = np.einsum("...ij,...j", precessionTissue, self.vecMArrayTissue[..., 0])[..., None]
                
                
                
                # The magnitude change due to relaxation is then applied to each
                # coordinate
                vecMIsochromat[:,:,:,0,:] *= exp_delta_t_t2_star_tissue
                vecMIsochromat[:,:,:,1,:] *= exp_delta_t_t2_star_tissue
                vecMIsochromat[:,:,:,2,:] = one_minus_exp_delta_t_t1_tissue + vecMIsochromat[:,:,:,2,:]*exp_delta_t_t1_tissue
                #The stored array is then updated
                self.vecMArrayTissue = vecMIsochromat
                

                #Multiply by the precession rotation matrix (incremental for each deltaT)
                #vecMIsochromat = np.matmul(precessionBlood, self.vecMArrayBlood)
                vecMIsochromat = np.einsum("...ij,...j", precessionBlood, self.vecMArrayBlood[..., 0])[..., None]

                
                # The magnitude change due to relaxation is then applied to each
                # coordinate
                vecMIsochromat[:,:,:,0,:] *= exp_delta_t_t2_star_blood
                vecMIsochromat[:,:,:,1,:] *= exp_delta_t_t2_star_blood
                vecMIsochromat[:,:,:,2,:] = one_minus_exp_delta_t_t1_blood + vecMIsochromat[:,:,:,2,:]*exp_delta_t_t1_blood
                #The stored array is then updated
                self.vecMArrayBlood = vecMIsochromat
                

                #Combine tissue and blood compartments to give the total magnetization 
                # vector array
                vecMArray = self.vecMArrayTissue
                vecMArray = np.concatenate((vecMArray,self.vecMArrayBlood),axis=0)

                #If the total time that has passed corresponds to the time at which
                # there is an echo peak:
                if int(totalTime/self.deltaT) in self.signalDivide and signal_flag == True: 
                    #Get the index of the peak (what number peak is it?)
                    signalDivide = list(self.signalDivide)
                    ind = signalDivide.index(int(totalTime/self.deltaT))
                    #Then input the magentization array at that time into the siganl
                    # holder array
                    try:
                        self.signal[:,0,:,:,ind] = np.squeeze(vecMArray)
                    except:
                        self.signal[:,0,:,:,ind] = np.expand_dims(np.squeeze(vecMArray), axis=1)

                    #print(f"totalTime: {totalTime}, signalDivide: {self.signalDivide[ind]}")
                

        return totalTime

    def MRFSGRE(self):   
        
        """
        Bloch Simulation code for a two compartment model with a semipermeable barrier 
        INCLUDING:
        - 2D set of isochromats with frequency encoding 
        - Exchange between two compartments
        - Variable fractional compartment sizes  

        IMPORTANT ASSUMPTIONS
        - The effects of relaxation are ignored during the RF pulses 
        - Between pulses all transverse magnetization has relaxed back to
        equilibrium
        - Frequencies in MHz have been converted to Hz to make the simulation
        easier to visualise
        
        Returns:
        --------
        signalNoisy : numpy nd array, shape (noOfRepetitions, noise)
            Noisy signal array (magnitude of magnetization at echo time for each noise level)
        """
    
        """Gradient parameters"""
        # Maximum gradient height
        magnitudeOfGradient  =-6e-3 #5.49e-3 FISP PAPER # FISP gradient: -(2e-3)*np.pi , SPGRE: -6e-3 #UNIT: T/m

        ### Set the duration of the gradients applied to have the echo peak occur at TE
        firstGradientDuration = self.TE/2
        secondGradientDuration = self.TE

        ### Set all time arrays to zero
        totalTime = 0

        '''
            INVERSION RECOVERY
        '''
        
        # THIS IS REDUNDANT, IT IS IN THE FA TRAIN
        """
        ### Include an inversion recovery pulse to null CSF :'0' - No, '1' - 'Yes'      
        if self.inv == 1 and self.CSF_null == True:

            # T1 of CSF set to 4500 ms from  Shin W, Gu H, Yang Y. 
            #Fast high-resolution T1 mapping using inversion-recovery Look–Locker
            #echo-planar imaging at steady state: optimization for accuracy and 
            #reliability. Magn Reson Med 2009;61(4):899–906. 
            t1CSF = 4500 #UNIT ms 
            # If TR>T1 then the formula: tNull = 0.69*T1 can be used 
            # For a SE sequence when TR!>T11 then: tNull = T1[ln2 - ln(1-e(-TR/T1))]
            # From Bernstein. Handbook of MRI pulse sequences. 2004. p631-5. 
            # I believe this is still applicable for a GE sequence
            # MRF patch does not allow for multiple TI values so use mean TR to
            # calcualte the TI
            tNull = 2909 #int(t1CSF*0.69) #np.log(2)-np.log(1-np.exp(-np.mean(trRound)/t1CSF)))

            THE ABOVE IS INCORRECT, should have TI = T1ln2 from Bernstein
        
            #Application of inversion pulse 
            self.invpulse(loop=0)
            
            # Set all gradients to zero
            self.gradientX = 0
            self.gradientY = 0
            
            # Allow time for inversion recovery to null CSF
            #variable timeChuck to hold the array we dont want
            timeChuck = self.longTR(int(tNull), totalTime)

        elif self.inv ==1 and self.sequence == 'FISP' and self.CSF_null == False:
            print('inverted')
            self.invpulse(loop=0)
            inv_time = 20
            timeChuck = self.longTR(inv_time, totalTime)
        """
            
            
        '''
            MAIN LOOP
        '''
        
        timeArray = np.zeros([int((self.perc*self.noOfIsochromatsX)/100), self.noOfIsochromatsY, self.noOfIsochromatsZ, 1])

        ### Loop over each TR repetitions
        for loop in range(self.noOfRepetitions):
            '''
            WATER EXCHANGE
            '''  
            # loop time added for each iteration
            repTime = self.trRound[loop]
            #Generate array for storing reseidence time for exchange isochromat
            timeArray = timeArray + np.tile(repTime, [int((self.perc*self.noOfIsochromatsX)/100), self.noOfIsochromatsY, self.noOfIsochromatsZ, 1])       
            # same number of random numbers found (one for each isochromat)
            rands = np.random.uniform(0,1,[int((self.perc*self.noOfIsochromatsX)/100), self.noOfIsochromatsY,self.noOfIsochromatsZ, 1])
            #cumulative probability of exchange is calculated for each isochromat 
            # at the given time point 
            cum = 1 - np.exp(-timeArray/self.res)

            # exchange points when the cumulative prpobability is greater than the 
            # random number 
            exch = rands - cum
            exch = (exch < 0)
            exch = exch*1
            indBlood = np.argwhere(exch == 1)[:,:3]
            # Chose random isochromat to exchange with
            randsX = np.random.randint(0, np.size(self.vecMArrayTissue,0), int(np.size(np.argwhere(exch == 1),0)))
            randsY = np.random.randint(0, np.size(self.vecMArrayTissue,1), int(np.size(np.argwhere(exch == 1),0)))
            randsZ = np.random.randint(0, np.size(self.vecMArrayTissue,2), int(np.size(np.argwhere(exch == 1),0)))


            # Swap
            for change in range(int(np.size(np.argwhere(exch == 1),0))):
                self.vecMArrayBlood[indBlood[change,0],indBlood[change,1],indBlood[change,2],:], self.vecMArrayTissue[randsX[change],randsY[change],randsZ[change],:] = \
                    self.vecMArrayTissue[randsX[change],randsY[change],randsZ[change],:], self.vecMArrayBlood[indBlood[change,0],indBlood[change,1],indBlood[change,2],:]  

                
            # reset time array
            reset = cum - rands
            reset = (reset < 0)
            reset = reset*1
            timeArray = timeArray * reset 
            
            
            
            if self.sequence == 'SPGRE':           
                perfect_spoil = True                
                ###    RF PULSE
                #Application of RF pulse modifying the vecM arrays
                self.rfpulse(loop)
                
                ### RF SPOILING 
                #Application of the rf spoiling modifying the vecM arrays
                if perfect_spoil == False:
                    self.rf_spoil(loop)
                
                #Apply first gradient lobe (to the edge of k-space)
                self.gradientX = -magnitudeOfGradient
                self.gradientY = magnitudeOfGradient  #+ 2*(1/noOfRepetitions)*loop*magnitudeOfGradient
            
                ## FIRST SHORT GRADIENT
                #Precession occuring during gradient application 
                #Accounts for relaxation and applies precession to the vecM
                totalTime = self.applied_precession(firstGradientDuration, totalTime)
                
                #Apply second gradient lobe (traversing k-space)       
                self.gradientX = magnitudeOfGradient
                self.gradientY = -magnitudeOfGradient
                
                ## SECOND GRADIENT - DOUBLE LENGTH
                #Precession occuring during gradient application 
                #Accounts for relaxation and applies precession to the vecM
                saveMxMy = False
                if saveMxMy == False or loop > 10:
                    totalTime = self.applied_precession(secondGradientDuration, totalTime)
                elif saveMxMy == True:
                    if loop == 0:
                        mxy_history_tissue = np.zeros([10, int(secondGradientDuration/self.deltaT)])
                        mxy_history_blood = np.zeros([10, int(secondGradientDuration/self.deltaT)])
                    while loop <= 10:
                        totalTime = self.applied_precession(secondGradientDuration, totalTime, saveMxMy, mxy_history_tissue, mxy_history_blood, loop)
       
                
                
                # Calculate time passed
                passed = (self.pulseDuration + firstGradientDuration + secondGradientDuration)
                
                #Turn off gradients for remaining TR to play out
                self.gradientX = 0
                self.gradientY = 0
            
                # Allow remaining TR time
                totalTime = self.longTR((self.trRound[loop]-passed), totalTime)

                if perfect_spoil == True: # to debug
                # PERFECT SPOILING: Set Mxy components to zero
                    self.vecMArrayTissue[:,:,:,0,:] = 0  # Set Mx to zero for tissue
                    self.vecMArrayTissue[:,:,:,1,:] = 0  # Set My to zero for tissue
                    self.vecMArrayBlood[:,:,:,0,:] = 0   # Set Mx to zero for blood
                    self.vecMArrayBlood[:,:,:,1,:] = 0   # Set My to zero for blood
                
            elif self.sequence == 'FISP':

                sampling = 'spiral'
                
                # Fist, set the T2_star array to equal the T2_array because T2_star isn't
                # relevant here, and all of the implementation on code relies on T2_star
                self.t2StarArray = self.t2Array
                
                # From https://cds.ismrm.org/protected/18MProceedings/PDFfiles/images/1274/ISMRM2018-001274_Fig3.png (FISP)
                # Apply the RF pulse for the current FA
                self.rfpulse(loop)

                if sampling == 'cartesian':
                    # Apply first post-pulse gradient
                    self.gradientX = magnitudeOfGradient
                    self.gradientY = -magnitudeOfGradient
                    # Precession during gradient
                    totalTime = self.applied_precession(firstGradientDuration, totalTime)

                    # Apply gradient for twice as long
                    self.gradientX = -magnitudeOfGradient
                    self.gradientY = magnitudeOfGradient
                    # Precession during gradient
                    totalTime = self.applied_precession(secondGradientDuration, totalTime)

                    # Apply crusher gradient
                    self.gradientX = -magnitudeOfGradient
                    self.gradientY = magnitudeOfGradient
                    # Precession during gradient
                    totalTime = self.applied_precession(firstGradientDuration, totalTime,signal_flag=False)

                    # Calculate time passed
                    passed = (self.pulseDuration + firstGradientDuration*2 + secondGradientDuration)
                    
                    # Allow remaining TR time
                    totalTime = self.longTR((self.trRound[loop]-passed), totalTime, signal_flag=False)

               
                if sampling == 'spiral':
                    # The Song (HYDRA2019) FISP
                
                        # FID after pulse for TE and then sample signal
                        totalTime = self.longTR(self.TE, totalTime, signal_flag = True)

                        # Allow relaxation and unbalanced 'spoiler gradient' 
                        passed = self.TE + self.pulseDuration
                        spoilerTime = self.TE/2 #self.TE/2
                        totalTime = self.longTR(self.trRound[loop]-passed-spoilerTime, totalTime, signal_flag = False)
                        self.gradientX = -magnitudeOfGradient
                        self.gradientY = magnitudeOfGradient
                        totalTime = self.applied_precession(spoilerTime, totalTime, signal_flag=False)
                
        

        '''
        ADD NOISE 
        '''
        #Randomly select noise samples from existing noise array 
        noiseSize = np.shape(self.noiseArray) # sample noise storage array [dynamics, ]
        noiseSamples = np.random.choice(int(noiseSize[0]),[self.noOfRepetitions*self.samples])
        addedNoise = self.noiseArray[noiseSamples,:,:self.noise]

        #Transpose to fix shape
        addedNoise = np.transpose(addedNoise, (2,0,1))

        #Expand arrays to account for noise levels and samples
        vecPeaks = np.expand_dims(self.signal,axis=5)
        vecPeaks = np.tile(vecPeaks, [self.noise])
        
            
        
        #Open noisy signal array, dependent if complex or magnitude data is sought after
        if self.complexFing == False:
            signalNoisy = np.zeros([self.noOfRepetitions,self.noise])
        elif self.complexFing == True:
            signalNoisy = np.zeros([self.noOfRepetitions,2])

        # Precompute constants
        scale_factor = self.noOfIsochromatsX * self.noOfIsochromatsY * self.noOfIsochromatsZ * (1 / (self.noOfIsochromatsX * self.noOfIsochromatsY))
        #For each requested noise level
        for samp in range(self.samples):
            if self.sliceProfileSwitch == 1:
                try:
                    signalNoisyX = (np.squeeze((np.sum(np.sum(np.sum(vecPeaks[:,:,:,0,:,:],axis=0),axis=0),axis=0))) + 
                                    scale_factor * np.transpose(addedNoise[:,self.noOfRepetitions * samp:self.noOfRepetitions* (samp+1),0])) # scale factor accounts for the slice profile
                    
                    signalNoisyY = (np.squeeze((np.sum(np.sum(np.sum(vecPeaks[:,:,:,1,:], axis=0), axis=0),axis=0)))
                    + scale_factor*np.transpose(addedNoise[:,self.noOfRepetitions * samp:self.noOfRepetitions * (samp+1),1]))
                    
                    if self.sliceProfileType == 'FISP' or self.sliceProfileType == 'SPGRE': # my slice profile
                        signalNoisyX += (np.squeeze((np.sum(np.sum(np.sum(vecPeaks[:,:,1:,0,:,:],axis=0),axis=0),axis=0))))
                        signalNoisyY += (np.squeeze((np.sum(np.sum(np.sum(vecPeaks[:,:,1:,1,:], axis=0), axis=0),axis=0))))
                    
                    if self.complexFing == False:
                        #Find the total magitude of M 
                        signalNoisy[:,:] = np.sqrt((signalNoisyX)**2 + (signalNoisyY)**2)
                    elif self.complexFing == True:
                        # return the complex signal
                        signalNoisy[:,0] = np.squeeze(signalNoisyX)
                        signalNoisy[:,1] = np.squeeze(signalNoisyY)

                except:
                    
                    signalNoisyX = (np.squeeze((np.sum(np.sum(np.sum(vecPeaks[:,:,:,0,:,:],axis=0),axis=0),axis=0)))
                    + (self.noOfIsochromatsX * self.noOfIsochromatsY*self.noOfIsochromatsZ*(1/self.noOfIsochromatsX*self.noOfIsochromatsY)) * (
                        addedNoise[:,self.noOfRepetitions * samp:self.noOfRepetitions*(samp+1),0]))
                    
                    signalNoisyY = (np.squeeze((np.sum(np.sum(np.sum(vecPeaks[:,:,:,1,:], axis=0), axis=0),axis=0))) 
                    + (self.noOfIsochromatsX * self.noOfIsochromatsY*self.noOfIsochromatsZ*(1/self.noOfIsochromatsX*self.noOfIsochromatsY))* (
                        addedNoise[:,self.noOfRepetitions*samp:self.noOfRepetitions*(samp+1),1]))
                    
                    if self.sliceProfileType == 'FISP' or self.sliceProfileType == 'SPGRE': # my slice profile:
                        signalNoisyX += (np.squeeze((np.sum(np.sum(np.sum(vecPeaks[:,:,1:,0,:,:],axis=0),axis=0),axis=0))))
                    
                        signalNoisyY += (np.squeeze((np.sum(np.sum(np.sum(vecPeaks[:,:,1:,1,:], axis=0), axis=0),axis=0))))
                    
                    if self.complexFing == False:
                        #Find the total magitude of M 
                        signalNoisy[:,:] = np.transpose(np.sqrt((signalNoisyX)**2 + (signalNoisyY)**2))
                    elif self.complexFing == True:
                        # return the complex signal
                        signalNoisy[:,0] = np.transpose(np.squeeze(signalNoisyX))
                        signalNoisy[:,1] = np.transpose(np.squeeze(signalNoisyY))

            
            elif self.sliceProfileSwitch == 0:
                print('no slice profile')
                try:
                    signalNoisyX = (np.squeeze((np.sum(np.sum(np.sum(vecPeaks[:,:,:,0,:,:],axis=0),axis=0),axis=0))) + 
                                    (self.noOfIsochromatsX*self.noOfIsochromatsY*self.noOfIsochromatsZ*(1/self.noOfIsochromatsX*self.noOfIsochromatsY))*
                                    np.transpose(addedNoise[:,self.noOfRepetitions * samp:self.noOfRepetitions* (samp+1),0]))
                    
                    signalNoisyY = (np.squeeze((np.sum(np.sum(np.sum(vecPeaks[:,:,:,1,:], axis=0), axis=0),axis=0)))
                    + (self.noOfIsochromatsX * self.noOfIsochromatsY*self.noOfIsochromatsZ*(1/self.noOfIsochromatsX*self.noOfIsochromatsY))*np.transpose(
                        addedNoise[:,self.noOfRepetitions * samp:self.noOfRepetitions * (samp+1),1]))

                    if self.complexFing == False:
                        #Find the total magitude of M 
                        signalNoisy[:,:] = np.sqrt((signalNoisyX)**2 + (signalNoisyY)**2)
                    elif self.complexFing == True:
                        # return the complex signal
                        signalNoisy[:,0] = np.squeeze(signalNoisyX)
                        signalNoisy[:,1] = np.squeeze(signalNoisyY)
                
                except:
                    signalNoisyX = (np.squeeze((np.sum(np.sum(np.sum(vecPeaks[:,:,:,0,:,:],axis=0),axis=0),axis=0)))
                    + (self.noOfIsochromatsX * self.noOfIsochromatsY*self.noOfIsochromatsZ*(1/self.noOfIsochromatsX*self.noOfIsochromatsY)) * (
                        addedNoise[:,self.noOfRepetitions * samp:self.noOfRepetitions*(samp+1),0]))
                    
                    signalNoisyY = (np.squeeze((np.sum(np.sum(np.sum(vecPeaks[:,:,:,1,:], axis=0), axis=0),axis=0))) 
                    + (self.noOfIsochromatsX * self.noOfIsochromatsY*self.noOfIsochromatsZ*(1/self.noOfIsochromatsX*self.noOfIsochromatsY))* (
                        addedNoise[:,self.noOfRepetitions*samp:self.noOfRepetitions*(samp+1),1]))
                    
                    #Find the total magitude of M 
                    if self.complexFing == False:
                        #Find the total magitude of M 
                        signalNoisy[:,:] = np.transpose(np.sqrt((signalNoisyX)**2 + (signalNoisyY)**2))
                    elif self.complexFing == True:
                        # return the complex signal
                        signalNoisy[:,0] = np.transpose(np.squeeze(signalNoisyX))
                        signalNoisy[:,1] = np.transpose(np.squeeze(signalNoisyY))
            

            #Save signal according to structure
            if self.filesavetype == 'npy':
                #signal save file name
                if self.sequence == 'SPGRE':
                    signalName = 'echo_' + str(int(self.t1Array[0]))+ '_' + str(int(self.t1Array[1])) + '_'  \
                    + str(int(self.res)) + '_' + str(self.perc) + '_' + str(self.multi) + '_'
                if self.sequence == 'FISP':
                    signalName = 'echo_' + str(self.t1Array[0]) + '_' + str(self.t1Array[1]) + '_'  + str(self.t2Array[0]) + '_'  + str(self.t2Array[1]) + '_'\
                    + str(self.res) + '_' + str(self.perc) + '_' + str(self.multi) + '_'
                name = '../dictionaries/Dictionary' + self.dictionaryId +'/' + signalName + str(samp + 1)
                np.save(name, signalNoisy)

            if self.filesavetype == 'hdf5' or 'None':
                return signalNoisy
        
        return signalNoisy