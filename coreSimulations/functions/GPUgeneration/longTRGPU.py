# -*- coding: utf-8 -*-
"""-----------------------------------------------------------------------------
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


