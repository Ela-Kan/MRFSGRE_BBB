""" Class for methods to generate a Magnetic Resonance Fingerprinting (MRF) dictionary using the GPU."""

""" PACKAGE IMPORTS """
import numpy as np
import sys
import os
import platform
import warnings
from scipy import signal, io


class DictionaryGeneratorFast():
    # Initialise the class
    def __init__(self, t1Array, t2Array, t2StarArray, noOfIsochromatsX,
                noOfIsochromatsY, noOfIsochromatsZ, noOfRepetitions, noise, perc, res,
                multi, inv, sliceProfileSwitch, samples, dictionaryId, instance):
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
        sliceProfileSwitch : bool
            Slice profile switch
        samples : int
            Number of noise samples generated (set to one for dictionary generation)
        dictionaryId : str
            Dictionary ID
        instance : int
            Instance number for multiple instances of the same code to run
        
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
        self.samples = samples
        self.dictionaryId = dictionaryId
        self.instance = instance

    
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
        self.deltaT = 1 #ms

        #Initially gradient is 0 (while pulse is on)
        self.gradientX = 0 #T/m
        self.gradientY = 0 #T/m
        
        

        ### Set echo time (must be divisible by deltaT) 
        ## TO DO: Need to remove hard coding for TE=2 out of calculation code 
        self.TE = 2 #ms   

        '''
        SLICE PROFILE ARRAY READ IN
        '''
        if sliceProfileSwitch == 1: 
            sliceProfilePath = '../sliceProfile/sliceProfile.mat'
            sliceProfileArray = io.loadmat(sliceProfilePath)['sliceProfile']
            #to give an even sample of the slice profile array 
            endPoint = np.size(sliceProfileArray, 1)
            stepSize = (np.size(sliceProfileArray, 1)/2)/(noOfIsochromatsZ)
            startPoint = (np.size(sliceProfileArray, 1)/2) #stepSize/2 
            profileSamples = np.arange(startPoint, endPoint, stepSize, dtype=int) #np.round(np.linspace(0+27, np.size(sliceProfileArray,1)-1-27, noOfIsochromatsZ),0)
            #profileSamples = profileSamples.astype(int)
            self.sliceProfile = sliceProfileArray[:,profileSamples]
        else: 
            #If you want flat slice profile then this (i.e. homogenous excitation profile across the slice thickness)
            self.sliceProfile = np.tile(np.expand_dims(np.round(np.linspace(1,90,90*100),0), axis=1),noOfIsochromatsZ)   

        """---------------------------OPEN ARRAYS-----------------------------"""
        ### This is defined as a unit vector along the z-axis
        vecM = np.float64([[0],[0],[1]])
        
        ## Define arrays for blood and tissue 
        self.vecMArrayBlood = np.tile(vecM.T, [int(perc*noOfIsochromatsX/100), noOfIsochromatsY, noOfIsochromatsZ, 1])
        self.vecMArrayTissue = np.tile(vecM.T, [int(noOfIsochromatsX-perc*noOfIsochromatsX/100), noOfIsochromatsY, noOfIsochromatsZ, 1] )
            

        ### Expand the dimensions for multiplication 
        self.vecMArrayTissue = np.expand_dims(self.vecMArrayTissue, axis=4)
        self.vecMArrayBlood = np.expand_dims(self.vecMArrayBlood, axis=4)

        ### FA array
        faString = './functions/holdArrays/faArray_' + str(instance) + '.npy'
        self.faArray = np.load(faString) 

        ### Open and round TR array 
        trString = './functions/holdArrays/trArray_' + str(instance) + '.npy'
        trArray = np.load(trString)
        # Rounding is required in order to assure that TR is divisable by deltaT
        self.trRound = np.round(trArray, 0)
        
        ### Open noise sample array
        self.noiseArray = np.load('./functions/holdArrays/noiseSamples.npy')
        
        ### Empty signal array to store all magnitization at all time points 
        self.signal = np.zeros([noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3, noOfRepetitions])


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
        thetaX = np.pi*self.multi*0.8*np.ones([self.noOfIsochromatsZ])
        
        rotX = np.zeros([len(thetaX),3,3])
        rotY = np.zeros([len(thetaX),3,3])
        #rotation (pulse) flips spins from aligned with the z-axis to
        #aligned with the x-axis
        #Rotates around the x axis
        for theta in range(len(thetaX)):
            rotX[theta,:,:] = np.array([[1, 0, 0], [0, np.cos(thetaX[theta]), np.sin(thetaX[theta])], \
                            [0, -np.sin(thetaX[theta]), np.cos(thetaX[theta])]])
            rotY[theta,:,:] = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
        vecMRotation = np.matmul(rotY,rotX) 

        # Updating the magnetization vector matricies
        #For tissue
        self.vecMArrayTissue = np.matmul(vecMRotation,self.vecMArrayTissue)
        #For blood 
        self.vecMArrayBlood = np.matmul(vecMRotation,self.vecMArrayBlood)

        return None
    
    def longTR(self, remainingDuration, totalTime):
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

            #Set a hold array
            vecMIsochromat = self.vecMArrayBlood

            # The magnitude change due to relaxation is then applied to each
            # coordinate
            vecMIsochromat[:,:,:,0,:] = np.exp(-(remainingDuration)/t2Star)*vecMIsochromat[:,:,:,0,:]
            vecMIsochromat[:,:,:,1,:] = np.exp(-(remainingDuration)/t2Star)*vecMIsochromat[:,:,:,1,:]
            vecMIsochromat[:,:,:,2,:] = (1-np.exp(-remainingDuration/t1))*1  + vecMIsochromat[:,:,:,2,:]*(np.exp(-remainingDuration/t1))

            #The stored array is then updated
            self.vecMArrayBlood = vecMIsochromat

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
        
        faInt = int(self.faArray[loop]*100)
        #Extract the flip angle of this loop (degrees)
        if faInt != 0:
            try: 
                fa = self.multi*self.sliceProfile[faInt-1,:]
            except: 
                fa = self.multi*np.ones([self.noOfIsochromatsZ])*180
        else: 
            fa = self.multi*np.zeros([self.noOfIsochromatsZ])*180
        
        #Convert to radians
        thetaX = ((fa/360)*2*np.pi)  
        
        rotX = np.zeros([len(thetaX),3,3])
        rotY = np.zeros([len(thetaX),3,3])
        #rotation (pulse) flips spins from aligned with the z-axis to
        #aligned with the x-axis
        #Rotates around the x axis  
        for theta in range(len(thetaX)):
            rotX[theta,:,:] = np.array([[1, 0, 0], [0, np.cos(thetaX[theta]), np.sin(thetaX[theta])], \
                            [0, -np.sin(thetaX[theta]), np.cos(thetaX[theta])]])
            rotY[theta,:,:] = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])

        #Combined rotation (in this case same as rotX)
        vecMRotation = np.matmul(rotY,rotX) 

        # Updating the magnetization vector matricies
        #For tissue
        self.vecMArrayTissue = np.matmul(vecMRotation, self.vecMArrayTissue)
        #For blood 
        self.vecMArrayBlood = np.matmul(vecMRotation, self.vecMArrayBlood)

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
        
        #Rotation matrices for this rotation
        rotX = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
        rotY = np.array([[np.cos(thetaZ), -np.sin(thetaZ), 0],\
                            [np.sin(thetaZ), np.cos(thetaZ), 0],\
                            [0, 0, 1]])
        #Combined rotation (in this case same as rotY)
        vecMIsochromatHold = np.matmul(rotY,rotX)
        # Updating the matrix so each time only the incremental rotation is
        # calculated. 
        vecMIsochromatHold = np.matmul(rotY,rotX)
            
        # Updating the magnetization vector matricies
        #For tissue
        self.vecMArrayTissue = np.matmul(vecMIsochromatHold, self.vecMArrayTissue)
        #For blood 
        self.vecMArrayBlood = np.matmul(vecMIsochromatHold,self.vecMArrayBlood)

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

        # Gyromagnetic ratio for proton 42.6 MHz/T
        omegaArray = np.repeat(np.expand_dims((42.6)*gradientMatrix, axis=2), self.noOfIsochromatsZ, axis=2)

        #for the precessions generate an array storing the 3x3 rotation matrix 
        #for each isochromat
        precession = np.zeros([np.size(self.positionArrayX,0), np.size(self.positionArrayY,1),self.noOfIsochromatsZ, 3,3])
        precession[:,:,:,2,2] = 1

        # compute the trigonometric functions for the rotation matrices
        cos_omega_deltaT = np.cos(omegaArray*self.deltaT)
        sin_omega_deltaT = np.sin(omegaArray*self.deltaT)
        precession[:,:,:,0,0] = cos_omega_deltaT
        precession[:,:,:,0,1] = -sin_omega_deltaT
        precession[:,:,:,1,0] = sin_omega_deltaT
        precession[:,:,:,1,1] = cos_omega_deltaT

        return precession
   
    def applied_precession(self, gradientDuration, totalTime):
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
        
        Returns:
        --------
        vecMArrayTissue : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3)
            Array of magnetization vectors for the tissue compartment
        vecMArrayBlood : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3)
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
        
        #For each time step
        for tStep in range(int(gradientDuration/self.deltaT)):
            
                #Update time passed
                totalTime = totalTime + self.deltaT
            
                #For the tissue compartment
                #Set the relavent T1 and T2*
                t1 = self.t1Array[0]
                t2Star = self.t2StarArray[0]
                
                #Multiply by the precession rotation matrix (incremental for each deltaT)
                vecMIsochromat = np.matmul(precessionTissue, self.vecMArrayTissue)

                # The magnitude change due to relaxation is then applied to each
                # coordinate
                vecMIsochromat[:,:,:,0,:] = np.exp((-self.deltaT)/t2Star)*vecMIsochromat[:,:,:,0,:]
                vecMIsochromat[:,:,:,1,:] = np.exp((-self.deltaT)/t2Star)*vecMIsochromat[:,:,:,1,:]
                vecMIsochromat[:,:,:,2,:] = (1-np.exp(-self.deltaT/t1))*1 + vecMIsochromat[:,:,:,2,:]*(np.exp(-self.deltaT/t1))
                #The stored array is then updated
                self.vecMArrayTissue = vecMIsochromat

                #For the blood compartment
                #Set the relavent T1 and T2*
                t1 = self.t1Array[1]
                t2Star = self.t2StarArray[1]
                
                #Multiply by the precession rotation matrix (incremental for each deltaT)
                vecMIsochromat = np.matmul(precessionBlood, self.vecMArrayBlood)

                # The magnitude change due to relaxation is then applied to each
                # coordinate
                vecMIsochromat[:,:,:,0,:] = vecMIsochromat[:,:,:,0,:]*np.exp((-self.deltaT)/t2Star)
                vecMIsochromat[:,:,:,1,:] = np.exp((-self.deltaT)/t2Star)*vecMIsochromat[:,:,:,1,:]
                vecMIsochromat[:,:,:,2,:] = (1-np.exp(-self.deltaT/t1))*1 + vecMIsochromat[:,:,:,2,:]*(np.exp(-self.deltaT/t1))
                #The stored array is then updated
                self.vecMArrayBlood= vecMIsochromat
                
                #Combine tissue and blood compartments to give the total magnetization 
                # vector array
                vecMArray = self.vecMArrayTissue
                vecMArray = np.concatenate((vecMArray,self.vecMArrayBlood),axis=0)

                #If the total time that has passed corresponds to the time at which
                # there is an echo peak:
                if int(totalTime/self.deltaT) in self.signalDivide: 
                    #Get the index of the peak (what number peak is it?)
                    signalDivide = list(self.signalDivide)
                    ind = signalDivide.index(int(totalTime/self.deltaT))
                    #Then input the magentization array at that time into the siganl
                    # holder array
                    self.signal[:,0,:,:,ind] = np.squeeze(vecMArray)

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
        magnitudeOfGradient  = -6e-3 #UNIT: T/m

        ### Set the duration of the gradients applied to have the echo peak occur at TE
        firstGradientDuration = self.TE/2
        secondGradientDuration = self.TE

        ### Set all time arrays to zero
        totalTime = 0

        '''
            INVERSION RECOVERY
        '''
        
        ### Include an inversion recovery pulse to null CSF :'0' - No, '1' - 'Yes'      
        if self.inv == 1:

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
        
            #Application of inversion pulse 
            self.invpulse(loop=0)
            
            # Set all gradients to zero
            self.gradientX = 0
            self.gradientY = 0
            
            # Allow time for inversion recovery to null CSF
            #variable timeChuck to hold the array we dont want
            timeChuck = self.longTR(int(tNull), totalTime)
            
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
                hold = self.vecMArrayBlood[indBlood[change,0],indBlood[change,1],indBlood[change,2],:]
                self.vecMArrayBlood[indBlood[change,0],indBlood[change,1],indBlood[change,2]] = self.vecMArrayTissue[randsX[change],randsY[change],randsZ[change],:]
                self.vecMArrayTissue[randsX[change],randsY[change],randsZ[change],:] = hold 
                
            # reset time array
            reset = cum - rands
            reset = (reset < 0)
            reset = reset*1
            timeArray = timeArray * reset 
                                
            ###    RF PULSE
            #Application of RF pulse modifying the vecM arrays
            self.rfpulse(loop)
            
            ### RF SPOILING 
            #Application of the rf spoiling modifying the vecM arrays
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
            totalTime = self.applied_precession(secondGradientDuration, totalTime)
            
            
            # Calculate time passed
            passed = (self.pulseDuration + firstGradientDuration + secondGradientDuration)
            
            #Turn off gradients for remaining TR to play out
            self.gradientX = 0
            self.gradientY = 0
        
            # Allow remaining TR time
            totalTime = self.longTR((self.trRound[loop]-passed), totalTime)
            
        '''
        ADD NOISE 
        '''

        
        #Randomly select noise samples from existing noise array 
        noiseSize = np.shape(self.noiseArray)
        noiseSamples = np.random.choice(int(noiseSize[0]),[self.noOfRepetitions*self.samples])
        addedNoise = self.noiseArray[noiseSamples,:,:self.noise]

        #Transpose to fix shape
        addedNoise = np.transpose(addedNoise, (2,0,1))
        
        #Expand arrays to account for noise levels and samples
        vecPeaks = np.expand_dims(self.signal,axis=5)
        vecPeaks = np.tile(vecPeaks, [self.noise])
        
        #signal save file name
        signalName = 'echo_' + str(self.t1Array[0]) + '_' + str(self.t1Array[1]) + '_'  \
        + str(self.res) + '_' + str(self.perc) + '_' + str(self.multi) + '_'
        
        #Open noisy signal array
        signalNoisy = np.zeros([self.noOfRepetitions,self.noise])

        #For each requested noise level
        for samp in range(self.samples):
            #Find the magnitude for the Mx and My components for the magnetization
            # vector and add noise
            try:
                signalNoisyX = (np.squeeze((np.sum(np.sum(np.sum(vecPeaks[:,:,:,0,:,:],axis=0),axis=0),axis=0))) + 
                                (self.noOfIsochromatsX*self.noOfIsochromatsY*self.noOfIsochromatsZ*(1/self.noOfIsochromatsX*self.noOfIsochromatsY))*
                                np.transpose(addedNoise[:,self.noOfRepetitions * samp:self.noOfRepetitions* (samp+1),0]))
                
                signalNoisyY = (np.squeeze((np.sum(np.sum(np.sum(vecPeaks[:,:,:,1,:], axis=0), axis=0),axis=0)))
                + (self.noOfIsochromatsX * self.noOfIsochromatsY*self.noOfIsochromatsZ*(1/self.noOfIsochromatsX*self.noOfIsochromatsY))*np.transpose(
                    addedNoise[:,self.noOfRepetitions * samp:self.noOfRepetitions * (samp+1),1]))

                #Find the total magitude of M 
                signalNoisy[:,:] = np.sqrt((signalNoisyX)**2 + (signalNoisyY)**2)
                
            except:
                signalNoisyX = (np.squeeze((np.sum(np.sum(np.sum(vecPeaks[:,:,:,0,:,:],axis=0),axis=0),axis=0)))
                + (self.noOfIsochromatsX * self.noOfIsochromatsY*self.noOfIsochromatsZ*(1/self.noOfIsochromatsX*self.noOfIsochromatsY)) * (
                    addedNoise[:,self.noOfRepetitions * samp:self.noOfRepetitions*(samp+1),0]))
                
                signalNoisyY = (np.squeeze((np.sum(np.sum(np.sum(vecPeaks[:,:,:,1,:], axis=0), axis=0),axis=0))) 
                + (self.noOfIsochromatsX * self.noOfIsochromatsY*self.noOfIsochromatsZ*(1/self.noOfIsochromatsX*self.noOfIsochromatsY))* (
                    addedNoise[:,self.noOfRepetitions*samp:self.noOfRepetitions*(samp+1),1]))
                
                #Find the total magitude of M 
                signalNoisy[:,:] = np.transpose(np.sqrt((signalNoisyX)**2 + (signalNoisyY)**2))
                
            #Save signal         
            name = '../dictionaries/Dictionary' + self.dictionaryId +'/' + signalName + str(samp + 1)
            np.save(name, signalNoisy)


        return signalNoisy