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


   #Set the relevant T1[tissue,blood] and T2*[tissue,blood] exponential decay constants
   exp_t2_star_tissue = np.exp(-(remainingDuration)/t2StarArray[0])
   exp_t1_tissue = (np.exp(-remainingDuration/t1Array[0]))
   one_minus_exp_t1_tissue = (1-np.exp(-remainingDuration/t1Array[0]))


   #Set a hold array
   vecMIsochromat = vecMArrayTissue

   # The magnitude change due to relaxation is then applied to each
   # coordinate
   vecMIsochromat[:,:,:,0,:] *= exp_t2_star_tissue
   vecMIsochromat[:,:,:,1,:] *= exp_t2_star_tissue
   vecMIsochromat[:,:,:,2,:] *= exp_t1_tissue
   vecMIsochromat[:,:,:,2,:] += one_minus_exp_t1_tissue

   #The stored array is then updated
   vecMArrayTissue = vecMIsochromat

   # TODO: FIX T2 STAR ITS INCORRECTLY LABELLED HERE (keep to compare to GT for now)
   #For the blood compartment
   #Set the relavent T1 and T2*
   exp_t2_star_blood = np.exp(-(remainingDuration)/t2StarArray[0]) # SHOULD BE 1 IN T2STARARRAY
   exp_t1_blood = (np.exp(-remainingDuration/t1Array[1]))
   one_minus_exp_t1_blood = (1-np.exp(-remainingDuration/t1Array[1]))
   
   #Set a hold array
   vecMIsochromat = vecMArrayBlood

   # The magnitude change due to relaxation is then applied to each
   # coordinate
   vecMIsochromat[:,:,:,0,:] *= exp_t2_star_blood
   vecMIsochromat[:,:,:,1,:] *= exp_t2_star_blood
   vecMIsochromat[:,:,:,2,:] *= exp_t1_blood
   vecMIsochromat[:,:,:,2,:] += one_minus_exp_t1_blood
   
   #The stored array is then updated
   vecMArrayBlood = vecMIsochromat
   
   return vecMArrayTissue, vecMArrayBlood, totalTime


