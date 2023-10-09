# -*- coding: utf-8 -*-
"""-----------------------------------------------------------------------------
Application of an RF pulse to a Bloch equation simulation with two compartments 

Author: Emma Thomson
Year: 2022
Institution: Centre for Medical Image Computing: University College London
Email: e.thomson.19@ucl.ac.uk
-------------------------------------------------------------------------------"""

"""-------------------------------PACKAGES--------------------------------------"""

import numpy as np

"""---------------------------MAIN FUNCTION--------------------------------------"""

def invpulse(vecMArrayTissue, vecMArrayBlood, loop, noOfIsochromatsZ, multi):
    
    #180 pulse for inversion
    #because the 180 pulse is rubbish multiply value by 0.7
    #this is extremely crude - need to add either another parameter 
    # or manually code the IR pulse in the sequence code
    thetaX = np.pi*multi*0.7*np.ones([noOfIsochromatsZ])
    
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
    vecMArrayTissue = np.matmul(vecMRotation,vecMArrayTissue)
    #For blood 
    vecMArrayBlood = np.matmul(vecMRotation,vecMArrayBlood)

    return vecMArrayTissue, vecMArrayBlood

