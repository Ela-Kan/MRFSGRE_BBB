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

def rf_spoil(vecMArrayTissue, vecMArrayBlood, loop):
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
    vecMArrayTissue = np.matmul(vecMIsochromatHold,vecMArrayTissue)
    #For blood 
    vecMArrayBlood = np.matmul(vecMIsochromatHold,vecMArrayBlood)

    return vecMArrayTissue, vecMArrayBlood
