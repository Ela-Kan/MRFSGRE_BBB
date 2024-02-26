# -*- coding: utf-8 -*-
"""-----------------------------------------------------------------------------
Calculation of the precession of isochormats during the application of gradients
for a two compartment model 

Relaxation is considered.

Author: Emma Thomson
Year: 2022
Institution: Centre for Medical Image Computing: University College London
Email: e.thomson.19@ucl.ac.uk
-------------------------------------------------------------------------------"""

"""-------------------------------PACKAGES--------------------------------------"""
import numpy as np 

"""---------------------------MAIN FUNCTION--------------------------------------"""

from rotationCalculations import rotation_calculations
   
def applied_precession(gradientDuration, deltaT, gradientX, gradientY,
    positionArrayX, positionArrayY, noOfIsochromatsZ,vecMArrayTissue,vecMArrayBlood,t1Array,
    t2StarArray, signal, totalTime, signalDivide):

   #Calculate the rotation matrices with different spatial matrices for each isochromat
   #in the array
   precession = rotation_calculations(positionArrayX,positionArrayY,
                 gradientX, gradientY, noOfIsochromatsZ, deltaT)
   
   
   #Transpose to correct shape
   precession = precession.transpose(1,0,2,4,3)

   
   #Separate the large precession array into the blood and tissue compartments
   precessionBlood = precession[:np.size(vecMArrayBlood,0),:, :, :]
   precessionTissue = precession[np.size(vecMArrayBlood,0):,:, :, :]


   
   #For each time step
   for tStep in range(int(gradientDuration/deltaT)):
       
        #Update time passed
        totalTime = totalTime + deltaT
    
        #For the tissue compartment
        #Set the relavent T1 and T2*
        t1 = t1Array[0]
        t2Star = t2StarArray[0]
        
        #Multiply by the precession rotation matrix (incremental for each deltaT)
        vecMIsochromat = np.matmul(precessionTissue, vecMArrayTissue)

        # The magnitude change due to relaxation is then applied to each
        # coordinate
        vecMIsochromat[:,:,:,0,:] = np.exp((-deltaT)/t2Star)*vecMIsochromat[:,:,:,0,:]
        vecMIsochromat[:,:,:,1,:] = np.exp((-deltaT)/t2Star)*vecMIsochromat[:,:,:,1,:]
        vecMIsochromat[:,:,:,2,:] = (1-np.exp(-deltaT/t1))*1 + vecMIsochromat[:,:,:,2,:]*(np.exp(-deltaT/t1))
        #The stored array is then updated
        vecMArrayTissue = vecMIsochromat

        #For the blood compartment
        #Set the relavent T1 and T2*
        t1 = t1Array[1]
        t2Star = t2StarArray[1]
        
        #Multiply by the precession rotation matrix (incremental for each deltaT)
        vecMIsochromat = np.matmul(precessionBlood, vecMArrayBlood)

        # The magnitude change due to relaxation is then applied to each
        # coordinate
        vecMIsochromat[:,:,:,0,:] = vecMIsochromat[:,:,:,0,:]*np.exp((-deltaT)/t2Star)
        vecMIsochromat[:,:,:,1,:] = np.exp((-deltaT)/t2Star)*vecMIsochromat[:,:,:,1,:]
        vecMIsochromat[:,:,:,2,:] = (1-np.exp(-deltaT/t1))*1 + vecMIsochromat[:,:,:,2,:]*(np.exp(-deltaT/t1))
        #The stored array is then updated
        vecMArrayBlood= vecMIsochromat
        
        #Combine tissue and blood compartments to give the total magnetization 
        # vector array
        vecMArray = vecMArrayTissue
        vecMArray = np.concatenate((vecMArray,vecMArrayBlood),axis=0)

        #If the total time that has passed corresponds to the time at which
        # there is an echo peak:
        if int(totalTime/deltaT) in signalDivide: 
            #Get the index of the peak (what number peak is it?)
            signalDivide = list(signalDivide)
            ind = signalDivide.index(int(totalTime/deltaT))
            #Then input the magentization array at that time into the siganl
            # holder array
            signal[:,0,:,:,ind] = np.squeeze(vecMArray)
         
   return vecMArrayTissue, vecMArrayBlood, signal, totalTime
