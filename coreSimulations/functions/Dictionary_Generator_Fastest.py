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
import warnings
from pathlib import Path


class Dictionary_Generator_Fastest():
    # Initialise the class
    def __init__(self, t1Array, t2Array, t2StarArray, noOfIsochromatsX,
                noOfIsochromatsY, noOfIsochromatsZ, noOfRepetitions, perc, res,
                multi, sliceProfileSwitch, sequence, instance, readTRFA= True, trArray = None, faArray = None):
        """
        Parameters:
        -----------
        For simulating water exhange. Inversion pulse should be added to the TR and FA train (e.g. TR[0]=TI and FA[0]=180.)

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
        perc : int
            Percentage blood volume 
        res : int
            intravascular water residence time UNIT: ms
        multi : float
            Multiplication factor for the B1 value
        sliceProfileSwitch : bool
            Slice profile switch
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
        self.perc = perc
        self.res = res
        self.multi = multi
        self.sequence = sequence
        self.instance = instance
        self.sliceProfileSwitch = sliceProfileSwitch

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
        self.complexFing = True



        """---------------------------OPEN ARRAYS-----------------------------"""
        ### This is defined as a unit vector along the z-axis
        vecM = np.float32([[0],[0],[1]])
        
        ## Define arrays for blood and tissue 
        self.numBloodIsosX = int(self.perc*self.noOfIsochromatsX/100)
        self.numTissueIsosX = self.noOfIsochromatsX - self.numBloodIsosX
        self.vecMArrayBlood = np.tile(vecM.T, [self.numBloodIsosX, self.noOfIsochromatsY, self.noOfIsochromatsZ, 1])
        self.vecMArrayTissue = np.tile(vecM.T, [self.numTissueIsosX, self.noOfIsochromatsY, self.noOfIsochromatsZ, 1] )
            

        ### Expand the dimensions for multiplication,  np.ascontiguousarray for speed
        self.vecMArrayTissue = np.ascontiguousarray(np.expand_dims(self.vecMArrayTissue, axis=4))
        self.vecMArrayBlood = np.ascontiguousarray(np.expand_dims(self.vecMArrayBlood, axis=4))

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

        self.integerFlip = True # flag for slice profile of original FISP paper select false
        if self.integerFlip and self.sliceProfileSwitch:
            self.fa_ints = (self.faArray * 100).astype(int) # for looking up in slice profile
            
        script_dir = Path(__file__).parent 

        
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
                project_root = script_dir.parent.parent
                sliceProfilePath =  str(project_root /'sliceProfile'/'sliceProfileRFSPGRE.mat')
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
            self.signalDivide[r] = (sum(self.trRound[:r])+self.TE+self.pulseDuration)/self.deltaT

        self.signalDivide_dict = {val: idx for idx, val in enumerate(self.signalDivide)}

        # holds for rotation matrix shapes
        self.rotX_template = np.zeros([self.noOfIsochromatsZ,3,3])
        
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
            vecMIsochromat[:,:,:,0,:] *= np.exp(-(remainingDuration)/t2Star)
            vecMIsochromat[:,:,:,1,:] *= np.exp(-(remainingDuration)/t2Star)
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
            vecMIsochromat[:,:,:,0,:] *=np.exp(-(remainingDuration)/t2Star)
            vecMIsochromat[:,:,:,1,:] *= np.exp(-(remainingDuration)/t2Star)
            vecMIsochromat[:,:,:,2,:] = (1-np.exp(-remainingDuration/t1))*1  + vecMIsochromat[:,:,:,2,:]*(np.exp(-remainingDuration/t1))

            #The stored array is then updated
            self.vecMArrayBlood = vecMIsochromat

            #If the total time that has passed corresponds to the time at which
            # there is an echo peak:

            vecMArray = np.concatenate((self.vecMArrayTissue,self.vecMArrayBlood),axis=0)
            time_index = int(totalTime/self.deltaT)
            #If the total time that has passed corresponds to the time at which
            # there is an echo peak:
            if signal_flag and time_index in self.signalDivide_dict:
                #Get the index of the peak (what number peak is it?)
                ind = self.signalDivide_dict[time_index]
                #Then input the magentization array at that time into the siganl
                # holder array
                try:
                    self.signal[:,0,:,:,ind] = np.squeeze(vecMArray)
                except:
                    self.signal[:,0,:,:,ind] = np.expand_dims(np.squeeze(vecMArray), axis=1)

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


        
        # Flag for integer flip angles
        if self.integerFlip == True and (self.sliceProfileType == 'FISP' or self.sliceProfileType == 'SPGRE'): # for my slice profile correction
            faInt = self.fa_ints[loop]
            
            #Extract the slice profile of the effective flip angle of this loop (degrees)
            if faInt != 0 and faInt != 180*100 :
                try: 
                    fa = self.multi*self.sliceProfile[faInt-1,:]

                except: 
                    print(f'B1 = {self.multi}. FA = {self.faArray[loop]}. Flip angle not in calculated slice profile. Using {180} degrees')
                    fa = np.ones([self.noOfIsochromatsZ])*180 #self.multi*np.ones([self.noOfIsochromatsZ])*180 
            
            elif faInt== 180*100: # for perfect inversion
                    fa = np.ones([self.noOfIsochromatsZ])*180
            else: 
                fa = self.multi*np.zeros([self.noOfIsochromatsZ])*180

        elif self.sliceProfileSwitch == 0 and self.integerFlip == False:
            fa = self.multi*np.ones([self.noOfIsochromatsZ])*(self.faArray[loop])


        elif self.sliceProfileSwitch == 0 and self.integerFlip == True:
            fa = self.multi*np.ones(self.noOfIsochromatsZ)*(self.faArray[loop])
            fa = fa.astype(int)
            
        
        #Convert to radians
        thetaX = np.deg2rad(fa) #((fa/360)*2*np.pi)
    
        
        
        sin_thetaX,  cos_thetaX = nbh.sincos(thetaX)
        self.rotX_template[:,0, 0] = 1
        self.rotX_template[:, 1, 1] = cos_thetaX
        self.rotX_template[:, 1, 2] = sin_thetaX
        self.rotX_template[:, 2, 1] = -sin_thetaX
        self.rotX_template[:, 2, 2] = cos_thetaX    

        # Updating the magnetization vector matrices
        #For tissue
        #self.vecMArrayTissue = np.matmul(rotX, self.vecMArrayTissue)
        self.vecMArrayTissue = np.einsum("...ij,...j", self.rotX_template, self.vecMArrayTissue[..., 0])[..., None]

        #For blood 
        #self.vecMArrayBlood = np.matmul(rotX, self.vecMArrayBlood)
        self.vecMArrayBlood = np.einsum("...ij,...j", self.rotX_template, self.vecMArrayBlood[..., 0])[..., None]

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

        return nbh.rotation_calculations_numba(self.gradientX, self.positionArrayX, self.gradientY, self.positionArrayY, self.noOfIsochromatsZ, self.deltaT)


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
        
        #Transpose to correct shape
        precession = precession.transpose(1,0,2,4,3)
        
        #Separate the large precession array into the blood and tissue compartments
        precessionBlood = precession[:np.size(self.vecMArrayBlood,0),:, :, :]
        precessionTissue = precession[np.size(self.vecMArrayBlood,0):,:, :, :]

        # Pre calculate the exponential multipliers
        #For the tissue compartment
        #Set the relevant T1 and T2*
        t1 = self.t1Array[0]
        t2Star = self.t2StarArray[0]
        exp_delta_t_t2_star_tissue = np.exp((-self.deltaT)/t2Star)
        exp_delta_t_t1_tissue = (np.exp(-self.deltaT/t1))
        one_minus_exp_delta_t_t1_tissue = (1-np.exp(-self.deltaT/t1))

        #For the blood compartment
        #Set the relevant T1 and T2*
        t1 = self.t1Array[1]
        t2Star = self.t2StarArray[1]
        exp_delta_t_t2_star_blood = np.exp((-self.deltaT)/t2Star)
        exp_delta_t_t1_blood = (np.exp(-self.deltaT/t1))
        one_minus_exp_delta_t_t1_blood = (1-np.exp(-self.deltaT/t1))
        
        #For each time step
        for tStep in range(int(gradientDuration/self.deltaT)):
            
                #Update time passed
                totalTime = totalTime + self.deltaT
                
                #Multiply by the precession rotation matrix (incremental for each deltaT)
                #vecMIsochromat = np.matmul(precessionTissue, self.vecMArrayTissue)
                vecMIsochromat = np.einsum("...ij,...j", precessionTissue, self.vecMArrayTissue[..., 0])[..., None]
                
                
                # The magnitude change due to relaxation is then applied to each
                # coordinate
                vecMIsochromat[:,:,:,0,:] *= exp_delta_t_t2_star_tissue
                vecMIsochromat[:,:,:,1,:] *= exp_delta_t_t2_star_tissue
                vecMIsochromat[:,:,:,2,:] = one_minus_exp_delta_t_t1_tissue + vecMIsochromat[:,:,:,2,:]*exp_delta_t_t1_tissue
                #The stored array is then updated
                self.vecMArrayTissue= vecMIsochromat
                

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
                vecMArray = np.concatenate((self.vecMArrayTissue,self.vecMArrayBlood),axis=0)
                time_index = int(totalTime/self.deltaT)
                #If the total time that has passed corresponds to the time at which
                # there is an echo peak:
                if signal_flag and time_index in self.signalDivide_dict:
                    #Get the index of the peak (what number peak is it?)
                    ind = self.signalDivide_dict[time_index]
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
        fingerprint : numpy nd array, shape (noOfRepetitions, 1)
            Signal array 
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
            MAIN LOOP
        '''
        
        timeArray = np.zeros([self.numBloodIsosX, self.noOfIsochromatsY, self.noOfIsochromatsZ, 1])

        ### Loop over each TR repetitions
        for loop in range(self.noOfRepetitions):
            '''
            WATER EXCHANGE
            '''  
            # loop time added for each iteration
            repTime = self.trRound[loop]
            #Generate array for storing reseidence time for exchange isochromat
            timeArray = timeArray + np.tile(repTime, [self.numBloodIsosX, self.noOfIsochromatsY, self.noOfIsochromatsZ, 1])       
            # same number of random numbers found (one for each isochromat)
            rands = np.random.uniform(0,1,[self.numBloodIsosX, self.noOfIsochromatsY,self.noOfIsochromatsZ, 1])
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
            """
            for change in range(int(np.size(np.argwhere(exch == 1),0))):
                self.vecMArrayBlood[indBlood[change,0],indBlood[change,1],indBlood[change,2],:], self.vecMArrayTissue[randsX[change],randsY[change],randsZ[change],:] = \
                    self.vecMArrayTissue[randsX[change],randsY[change],randsZ[change],:], self.vecMArrayBlood[indBlood[change,0],indBlood[change,1],indBlood[change,2],:]  
            """
            if len(indBlood) > 0: # if we have exchange
                blood_hold = self.vecMArrayBlood[indBlood[:,0], indBlood[:,1], indBlood[:,2], :].copy()
                self.vecMArrayBlood[indBlood[:,0], indBlood[:,1], indBlood[:,2], :] = self.vecMArrayTissue[randsX, randsY, randsZ, :]
                self.vecMArrayTissue[randsX, randsY, randsZ, :] = blood_hold

            # reset time array
        
            timeArray *= (cum <= rands).astype(int)
            
            
            if self.sequence == 'SPGRE':           
                perfect_spoil = False                
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
                
        fingerprint = np.zeros([self.noOfRepetitions, 2])
        
        # Sum across all isochromats to get the signal
        if self.sliceProfileSwitch == 1:
            # Sum across x, y, z dimensions for Mx and My components
            fingerprintX = np.sum(self.signal[:, :, :, 0, :], axis=(0, 1, 2))
            fingerprintY = np.sum(self.signal[:, :, :, 1, :], axis=(0, 1, 2))
            
            # For FISP/SPGRE slice profiles
            if self.sliceProfileType == 'FISP' or self.sliceProfileType == 'SPGRE':
                fingerprintX += np.sum(self.signal[:, :, 1:, 0, :], axis=(0, 1, 2))
                fingerprintY += np.sum(self.signal[:, :, 1:, 1, :], axis=(0, 1, 2))
            
            fingerprint[:, 0] = fingerprintX
            fingerprint[:, 1] = fingerprintY
            
        elif self.sliceProfileSwitch == 0:
            fingerprintX = np.sum(self.signal[:, :, :, 0, :], axis=(0, 1, 2))
            fingerprintY = np.sum(self.signal[:, :, :, 1, :], axis=(0, 1, 2))
            fingerprint[:, 0] = fingerprintX
            fingerprint[:, 1] = fingerprintY
        
        # If magnitude fingerprint is needed instead of complex:
        if self.complexFing == False:
            magnitude = np.sqrt(fingerprint[:, 0]**2 + fingerprint[:, 1]**2)
            fingerprint = magnitude.reshape(-1, 1)  # Keep 2D shape


        return fingerprint