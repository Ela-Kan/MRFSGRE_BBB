# -*- coding: utf-8 -*-
"""-----------------------------------------------------------------------------
Author: Emma Thomson, Ela Kanani
Year: 2022-2024
Institution: Centre for Medical Image Computing: University College London
Email: e.thomson.19@ucl.ac.uk
-------------------------------------------------------------------------------"""

"""-------------------------------PACKAGES--------------------------------------"""
import numpy as np

"""---------------------------MAIN FUNCTION--------------------------------------"""

def rotation_calculations(positionArrayX, positionArrayY, gradientX, \
    gradientY, noOfIsochromatsZ, deltaT):
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
    gradientMatrix = gradientX*positionArrayX 
    gradientMatrix += gradientY*positionArrayY

    # Gyromagnetic ratio for proton 42.6 MHz/T
    omegaArray = np.repeat(np.expand_dims((42.6)*gradientMatrix, axis=2), noOfIsochromatsZ, axis=2)

    #for the precessions generate an array storing the 3x3 rotation matrix 
    #for each isochromat
    precession = np.zeros([np.size(positionArrayX,0), np.size(positionArrayY,1),noOfIsochromatsZ, 3,3])
    precession[:,:,:,2,2] = 1

    # compute the trigonometric functions for the rotation matrices
    cos_omega_deltaT = np.cos(omegaArray*deltaT)
    sin_omega_deltaT = np.sin(omegaArray*deltaT)
    precession[:,:,:,0,0] = cos_omega_deltaT
    precession[:,:,:,0,1] = -sin_omega_deltaT
    precession[:,:,:,1,0] = sin_omega_deltaT
    precession[:,:,:,1,1] = cos_omega_deltaT

    return precession
