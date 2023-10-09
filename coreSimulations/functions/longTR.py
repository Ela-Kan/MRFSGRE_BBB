# -*- coding: utf-8 -*-
"""-----------------------------------------------------------------------------
Calculation of the relaxation that occurs as the remaining TR plays out for a 
group of isochromats in a two compartment model 

Author: Emma Thomson
Year: 2022
Institution: Centre for Medical Image Computing: University College London
Email: e.thomson.19@ucl.ac.uk
-------------------------------------------------------------------------------"""

"""-------------------------------PACKAGES--------------------------------------"""
import numpy as np 

"""---------------------------MAIN FUNCTION--------------------------------------"""
   
def longTR(remainingDuration, deltaT, gradientX, gradientY,positionArrayX, 
           positionArrayY,vecMArrayTissue,vecMArrayBlood,t1Array,t2StarArray,
           totalTime):
  
   #Update time passed 
   totalTime = totalTime + remainingDuration

   #For the tissue compartment
   #Set the relavent T1 and T2*
   t1 = t1Array[0]
   t2Star = t2StarArray[0]

    #Set a hold array
   vecMIsochromat = vecMArrayTissue

   # The magnitude change due to relaxation is then applied to each
   # coordinate
   vecMIsochromat[:,:,:,0,:] = np.exp(-(remainingDuration)/t2Star)*vecMIsochromat[:,:,:,0,:]
   vecMIsochromat[:,:,:,1,:] = np.exp(-(remainingDuration)/t2Star)*vecMIsochromat[:,:,:,1,:]
   vecMIsochromat[:,:,:,2,:] = (1-np.exp(-remainingDuration/t1))*1 + vecMIsochromat[:,:,:,2,:]*(np.exp(-remainingDuration/t1))
   #The stored array is then updated
   vecMArrayTissue = vecMIsochromat

   #For the blood compartment
   #Set the relavent T1 and T2*
   t1 = t1Array[1]
   
   #Set a hold array
   vecMIsochromat = vecMArrayBlood

   # The magnitude change due to relaxation is then applied to each
   # coordinate
   vecMIsochromat[:,:,:,0,:] = np.exp(-(remainingDuration)/t2Star)*vecMIsochromat[:,:,:,0,:]
   vecMIsochromat[:,:,:,1,:] = np.exp(-(remainingDuration)/t2Star)*vecMIsochromat[:,:,:,1,:]
   vecMIsochromat[:,:,:,2,:] = (1-np.exp(-remainingDuration/t1))*1  + vecMIsochromat[:,:,:,2,:]*(np.exp(-remainingDuration/t1))
   
   #The stored array is then updated
   vecMArrayBlood = vecMIsochromat
   
   return vecMArrayTissue, vecMArrayBlood, totalTime


