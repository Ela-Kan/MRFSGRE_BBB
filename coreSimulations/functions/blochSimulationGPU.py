# -*- coding: utf-8 -*-
"""-----------------------------------------------------------------------------
Bloch Simulaton code for a two compartment model with a semipermeable barrier 

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

Author: Emma Thomson
Year: 2022
Institution: Centre for Medical Image Computing: University College London
Email: e.thomson.19@ucl.ac.uk
-------------------------------------------------------------------------------"""

"""-------------------------------PACKAGES--------------------------------------"""

### IMPORT DEPENDENCIES
import numpy as np
from rfPulse import rfpulse
from appliedPrecession import applied_precession
from rfSpoil import rf_spoil
from longTR import longTR
from invPulse import invpulse
import platform
from scipy import signal, io
import os

"""---------------------------MAIN FUNCTION--------------------------------------"""
def MRFSGRE(t1Array, t2Array, t2StarArray, noOfIsochromatsX,
            noOfIsochromatsY, noOfIsochromatsZ, noOfRepetitions, noise, perc, res,
            multi, inv, sliceProfileSwitch, samples, dictionaryId, instance):   
 
    """--------------------PARAMETER DECLERATION------------------------------"""
    ### This is defined as a unit vector along the z-axis
    vecM = np.float64([[0],[0],[1]])
    
    ### The modifiable variables are set as followed:
    # Maximum gradient height
    magnitudeOfGradient  = -6e-3 #UNIT: T/m
    
    ### set the rf pulse duration
    ### TO DO: need to allow for variable slice profile and pulse duration
    pulseDuration = 0 #UNIT ms 
    
    ### calculated position array used for the prcession according to spatial gradients     
    [positionArrayXHold ,positionArrayYHold] = \
                np.meshgrid(range(noOfIsochromatsX),range(noOfIsochromatsY))
    positionArrayX = positionArrayXHold - ((noOfIsochromatsX/2))
    positionArrayY = positionArrayYHold - ((noOfIsochromatsY/2))
    
    
    ### Time increment
    deltaT = 1 #ms

    #Initially gradient is 0 (while pulse is on)
    gradientX = 0 #T/m
    gradientY = 0 #T/m 
    
    ### Set echo time (must be divisible by deltaT) 
    ## TO DO: Need to remove hard coding for TE=2 out of calculation code 
    TE = 2 #ms
    
    '''
        SLICE PROFILE ARRAY READ IN
    '''
    if sliceProfileSwitch == 1: 
        sliceProfilePath = './sliceProfile/sliceProfile.mat'
        sliceProfileArray = io.loadmat(sliceProfilePath)['sliceProfile']
        #to give an even sample of the slice profile array 
        endPoint = np.size(sliceProfileArray, 1)
        stepSize = (np.size(sliceProfileArray, 1)/2)/(noOfIsochromatsZ)
        startPoint = (np.size(sliceProfileArray, 1)/2) #stepSize/2 
        profileSamples = np.arange(startPoint, endPoint, stepSize, dtype=int) #np.round(np.linspace(0+27, np.size(sliceProfileArray,1)-1-27, noOfIsochromatsZ),0)
        #profileSamples = profileSamples.astype(int)
        sliceProfile = sliceProfileArray[:,profileSamples]
    else: 
    #If you want flat slice profile then this 
        sliceProfile = np.tile(np.expand_dims(np.round(np.linspace(1,90,90*100),0), axis=1),noOfIsochromatsZ)
    
    """---------------------------OPEN ARRAYS-----------------------------"""
    
    ### Define arrays for blood and tissue 
    vecMArrayBlood = np.tile(vecM.T, [int(perc*noOfIsochromatsX/100), noOfIsochromatsY, noOfIsochromatsZ, 1])
    vecMArrayTissue = np.tile(vecM.T, [int(noOfIsochromatsX-perc*noOfIsochromatsX/100), noOfIsochromatsY, noOfIsochromatsZ, 1] )
        
    ### Expand the dimensions for multiplication 
    vecMArrayTissue = np.expand_dims(vecMArrayTissue, axis=4)
    vecMArrayBlood = np.expand_dims(vecMArrayBlood, axis=4)
    
    ### FA array
    faString = './coreSimulations/functions/holdArrays/faArray_' + str(instance) + '.npy'
    print(os.cwd())
    faArray = np.load(faString) 

    ### Open and round TR array 
    trString = './coreSimulations/functions/holdArrays/trArray_' + str(instance) + '.npy'
    trArray = np.load(trString)
    # Rounding is required in order to assure that TR is divisable by deltaT
    trRound = np.round(trArray, 0)
    
    ### Open noise sample array
    noiseArray = np.load('./coreSimulations/functions//holdArrays/noiseSamples.npy')
    
    ### Empty signal array to store all magnitization at all time points 
    signal = np.zeros([noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3, noOfRepetitions])
    
    ### Set all time arrays to zero
    totalTime = 0
    timeArray = np.zeros([int((perc*noOfIsochromatsX)/100), noOfIsochromatsY, noOfIsochromatsZ, 1])
   
    ### Set the duration of the gradients applied to have the echo peak occur at TE
    firstGradientDuration = TE/2
    secondGradientDuration = TE
    
    ### Include an inversion recovery pulse to null CSF :'0' - No, '1' - 'Yes'
    if inv == 1:     
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
    
    '''
        PEAK LOCATIONS
    '''
    
    ### Calculate the echo peak position peak position
    signalDivide = np.zeros([noOfRepetitions])
    for r in range(noOfRepetitions):  
        if inv == 0:
            signalDivide[r] = (sum(trRound[:r])+TE+pulseDuration)/deltaT
        else: 
            signalDivide[r] = (sum(trRound[:r])+TE+pulseDuration)/deltaT 
     
            
    '''
        INVERSION RECOVERY
    '''
    
    
    if inv == 1:
        
        #Application of inversion pulse 
        [vecMArrayTissue, vecMArrayBlood] = invpulse(vecMArrayTissue, vecMArrayBlood, 0,noOfIsochromatsZ, multi)
        
        # Set all gradients to zero
        gradientX = 0
        gradientY = 0
        
        # Allow time for inversion recovery to null CSF
        #variable timeChuck to hold the array we dont want
        [vecMArrayTissue, vecMArrayBlood, timeChuck] = \
        longTR(int(tNull), deltaT, gradientX, gradientY,positionArrayX, 
             positionArrayY,vecMArrayTissue,vecMArrayBlood,t1Array,t2StarArray,
             totalTime)
    '''
        MAIN LOOP
    '''
    
    ### Loop over each TR repetitions
    for loop in range(noOfRepetitions):
        
        '''
           WATER EXCHANGE
        '''  
        
        # loop time added for each iteration
        repTime = trRound[loop]
        #Generate array for storing reseidence time for exchange isochromat
        timeArray = timeArray + np.tile(repTime, [int((perc*noOfIsochromatsX)/100), noOfIsochromatsY, noOfIsochromatsZ, 1])       
        # same number of random numbers found (one for each isochromat)
        rands = np.random.uniform(0,1,[int((perc*noOfIsochromatsX)/100), noOfIsochromatsY,noOfIsochromatsZ, 1])
        #cumulative probability of exchange is calculated for each isochromat 
        # at the given time point 
        cum = 1 - np.exp(-timeArray/res)

        # exchange points when the cumulative prpobability is greater than the 
        # random number 
        exch = rands - cum
        exch = (exch < 0)
        exch = exch*1
        indBlood = np.argwhere(exch == 1)[:,:3]
        # Chose random isochromat to exchange with
        randsX = np.random.randint(0, np.size(vecMArrayTissue,0), int(np.size(np.argwhere(exch == 1),0)))
        randsY = np.random.randint(0, np.size(vecMArrayTissue,1), int(np.size(np.argwhere(exch == 1),0)))
        randsZ = np.random.randint(0, np.size(vecMArrayTissue,2), int(np.size(np.argwhere(exch == 1),0)))
        # Swap
        for change in range(int(np.size(np.argwhere(exch == 1),0))):
            hold = vecMArrayBlood[indBlood[change,0],indBlood[change,1],indBlood[change,2],:]
            vecMArrayBlood[indBlood[change,0],indBlood[change,1],indBlood[change,2]] = vecMArrayTissue[randsX[change],randsY[change],randsZ[change],:]
            vecMArrayTissue[randsX[change],randsY[change],randsZ[change],:] = hold 
            
        # reset time array
        reset = cum - rands
        reset = (reset < 0)
        reset = reset*1
        timeArray = timeArray * reset 
                              
        ###    RF PULSE
        #Application of RF pulse modifying the vecM arrays
        [vecMArrayTissue, vecMArrayBlood] = rfpulse(vecMArrayTissue, vecMArrayBlood, loop, faArray, noOfIsochromatsZ, sliceProfile, multi)
        
        ### RF SPOILING 
        #Application of the rf spoiling modifying the vecM arrays
        [vecMArrayTissue, vecMArrayBlood] = rf_spoil(vecMArrayTissue, vecMArrayBlood, loop)
        
        #Apply first gradient lobe (to the edge of k-space)
        gradientX = -magnitudeOfGradient
        gradientY = magnitudeOfGradient  #+ 2*(1/noOfRepetitions)*loop*magnitudeOfGradient
    
        ## FIRST SHORT GRADIENT
        #Precession occuring during gradient application 
        #Accounts for relaxation
        [vecMArrayTissue, vecMArrayBlood, signal, totalTime] = \
           applied_precession(firstGradientDuration, deltaT, gradientX, gradientY,
                positionArrayX, positionArrayY,noOfIsochromatsZ,vecMArrayTissue,vecMArrayBlood,t1Array,
                t2StarArray, signal, totalTime, signalDivide)
           
        #Apply second gradient lobe (traversing k-space)       
        gradientX = magnitudeOfGradient
        gradientY = -magnitudeOfGradient
    
        
        ## SECOND GRADIENT - DOUBLE LENGTH
        #Precession occuring during gradient application 
        #Accounts for relaxation
        [vecMArrayTissue, vecMArrayBlood, signal, totalTime] = \
           applied_precession(secondGradientDuration, deltaT, gradientX, gradientY,
                positionArrayX, positionArrayY,noOfIsochromatsZ,vecMArrayTissue,vecMArrayBlood,t1Array,
                t2StarArray, signal, totalTime, signalDivide)
         
          
        # Calculate time passed
        passed = (pulseDuration + firstGradientDuration + secondGradientDuration)
        
        #Turn off gradients for remaining TR to play out
        gradientX = 0
        gradientY = 0
    
        # Allow remaining TR time
        [vecMArrayTissue, vecMArrayBlood, totalTime] = \
            longTR((trRound[loop]-passed), deltaT, gradientX, gradientY,positionArrayX, 
                 positionArrayY,vecMArrayTissue,vecMArrayBlood,t1Array,t2StarArray,
                 totalTime)
        
    '''
       ADD NOISE 
    '''

    
    #Randomly select noise samples from existing noise array 
    noiseSize = np.shape(noiseArray)
    noiseSamples = np.random.choice(int(noiseSize[0]),[noOfRepetitions*samples])
    addedNoise = noiseArray[noiseSamples,:,:noise]

    #Transpose to fix shape
    addedNoise = np.transpose(addedNoise, (2,0,1))
    
    #Expand arrays to account for noise levels and samples
    vecPeaks = np.expand_dims(signal,axis=5)
    vecPeaks = np.tile(vecPeaks, [noise])
    
    #signal save file name
    signalName = 'echo_' + str(t1Array[0]) + '_' + str(t1Array[1]) + '_'  \
    + str(res) + '_' + str(perc) + '_' + str(multi) + '_'
    
    #Open noisy signal array
    signalNoisy = np.zeros([noOfRepetitions,noise])

    #For each requested noise level
    for samp in range(samples):
        #Find the magnitude for the Mx and My components for the magnetization
        # vector and add noise
        try:
            signalNoisyX = (np.squeeze((np.sum(np.sum(np.sum(vecPeaks[:,:,:,0,:,:],axis=0),axis=0),axis=0))) + 
                            (noOfIsochromatsX*noOfIsochromatsY*noOfIsochromatsZ*(1/noOfIsochromatsX*noOfIsochromatsY))*
                            np.transpose(addedNoise[:,noOfRepetitions * samp:noOfRepetitions* (samp+1),0]))
            
            signalNoisyY = (np.squeeze((np.sum(np.sum(np.sum(vecPeaks[:,:,:,1,:], axis=0), axis=0),axis=0)))
            + (noOfIsochromatsX * noOfIsochromatsY*noOfIsochromatsZ*(1/noOfIsochromatsX*noOfIsochromatsY))*np.transpose(
                addedNoise[:,noOfRepetitions * samp:noOfRepetitions * (samp+1),1]))

            #Find the total magitude of M 
            signalNoisy[:,:] = np.sqrt((signalNoisyX)**2 + (signalNoisyY)**2)
            
        except:
            signalNoisyX = (np.squeeze((np.sum(np.sum(np.sum(vecPeaks[:,:,:,0,:,:],axis=0),axis=0),axis=0)))
            + (noOfIsochromatsX * noOfIsochromatsY*noOfIsochromatsZ*(1/noOfIsochromatsX*noOfIsochromatsY)) * (
                addedNoise[:,noOfRepetitions * samp:noOfRepetitions*(samp+1),0]))
            
            signalNoisyY = (np.squeeze((np.sum(np.sum(np.sum(vecPeaks[:,:,:,1,:], axis=0), axis=0),axis=0))) 
            + (noOfIsochromatsX * noOfIsochromatsY*noOfIsochromatsZ*(1/noOfIsochromatsX*noOfIsochromatsY))* (
                addedNoise[:,noOfRepetitions*samp:noOfRepetitions*(samp+1),1]))
            
            #Find the total magitude of M 
            signalNoisy[:,:] = np.transpose(np.sqrt((signalNoisyX)**2 + (signalNoisyY)**2))
            
        #Save signal         
        name = './Dictionaries/Dictionary' + dictionaryId +'/' + signalName + str(samp + 1)
        np.save(name, signalNoisy)

    return signalNoisy

