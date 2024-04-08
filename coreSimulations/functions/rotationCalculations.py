# -*- coding: utf-8 -*-
"""-----------------------------------------------------------------------------
Calculation of the rotation matrices for each isochromat in a two compartment 
model when gradients are applied leaving to spatially different gradient field 
strengths

Author: Emma Thomson
Year: 2022
Institution: Centre for Medical Image Computing: University College London
Email: e.thomson.19@ucl.ac.uk
-------------------------------------------------------------------------------"""

"""-------------------------------PACKAGES--------------------------------------"""
import numpy as np
import numba as nb


"""---------------------------MAIN FUNCTION--------------------------------------"""

@nb.jit(nopython=True)
def rotation_calculations(positionArrayX, positionArrayY, gradientX, \
    gradientY, noOfIsochromatsZ, deltaT):
    
    #Find gradient field strength from both x and y gradients at each isochromat 
    # position
    gradientMatrix = gradientX*positionArrayX 
    gradientMatrix += gradientY*positionArrayY

    # Gyromagnetic ratio for proton 42.6 MHz/T
    #omegaArray = np.repeat(np.expand_dims((42.6)*gradientMatrix, axis=2), noOfIsochromatsZ, axis=2)
    omegaArray = np.expand_dims((42.6)*gradientMatrix, axis=2)
    omegaArray = omegaArray.repeat(noOfIsochromatsZ).reshape(omegaArray.shape[1:], noOfIsochromatsZ)

    #for the precessions generage an array storing the 3x3 rotation matrix 
    #for each isochromat
    precession = np.zeros([np.size(positionArrayX,0), np.size(positionArrayY,1),noOfIsochromatsZ, 3,3])

    cos_omega_deltaT = np.cos(omegaArray*deltaT)
    sin_omega_deltaT = np.sin(omegaArray*deltaT)
    
    precession[:,:,:,2,2] = 1
    precession[:,:,:,0,0] = cos_omega_deltaT
    precession[:,:,:,0,1] = -sin_omega_deltaT
    precession[:,:,:,1,0] = sin_omega_deltaT
    precession[:,:,:,1,1] = cos_omega_deltaT = np.cos(omegaArray*deltaT)
    return precession
