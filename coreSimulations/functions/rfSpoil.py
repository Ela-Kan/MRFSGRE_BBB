# -*- coding: utf-8 -*-
"""-----------------------------------------------------------------------------
Application of an RF spoiling to a Bloch equation simulation with two compartments 

Author: Emma Thomson and Ela Kanani
Year: 2022-2024
Institution: Centre for Medical Image Computing: University College London
Email: e.thomson.19@ucl.ac.uk
-------------------------------------------------------------------------------"""

"""-------------------------------PACKAGES--------------------------------------"""
import numpy as np

"""---------------------------MAIN FUNCTION--------------------------------------"""

def rf_spoil(vecMArrayTissue, vecMArrayBlood, loop):

    
    #Using the rf phase formula developed by Zur et al (1991)
    # calculate the phase change for this particular repetition
    alpha0 = (123/360)*2*np.pi
    thetaZ = 0.5*alpha0*(loop**2+loop+2)
    cos_thetaZ = np.cos(thetaZ)
    sin_thetaZ = np.sin(thetaZ)
    
    #Rotation matrices for this rotation
    #rotX = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]]) identity so not needed
    rotY = np.array([[cos_thetaZ, -sin_thetaZ, 0],\
                        [sin_thetaZ, cos_thetaZ, 0],\
                        [0, 0, 1]])
    
    """ identity multiplications
    #Combined rotation (in this case same as rotY)
    vecMIsochromatHold = np.matmul(rotY,rotX)
    # Updating the matrix so each time only the incremental rotation is
    # calculated. 
    vecMIsochromatHold = np.matmul(rotY,rotX)
    """
         
    # Updating the magnetization vector matricies
    #For tissue
    vecMArrayTissue = np.matmul(rotY,vecMArrayTissue)
    #For blood 
    vecMArrayBlood = np.matmul(rotY,vecMArrayBlood)

    return vecMArrayTissue, vecMArrayBlood
