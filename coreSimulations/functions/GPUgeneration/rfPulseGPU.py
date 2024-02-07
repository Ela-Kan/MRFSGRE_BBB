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

def rfpulse(vecMArrayTissue, vecMArrayBlood, loop, faArray,noOfIsochromatsZ, sliceProfile, multi):
    """
    Application of an RF pulse to a Bloch equation simulation with two compartments.

    Parameters:
    -----------
    vecMArrayTissue : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3)
        Array of magnetization vectors for the tissue compartment
    vecMArrayBlood : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3)
        Array of magnetization vectors for the blood compartment
    loop : int
        Loop number in number of repetitions
    faArray : numpy nd array, shape (noOfRepetitions,)
        Array of flip angles
    noOfIsochromatsZ : int
        Number of isochromats in the z direction
    sliceProfile : numpy nd array, shape (9000, noOfIsochromatsZ)
        Slice profile array
    multi : float
        Multiplication factor for the B1 value

    Returns:
    --------
    vecMArrayTissue : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3)
        Array of magnetization vectors for the tissue compartment
    vecMArrayBlood : numpy nd array, shape (noOfIsochromatsX, noOfIsochromatsY, noOfIsochromatsZ, 3)
        Array of magnetization vectors for the blood compartment
    
    """
    
    faInt = int(faArray[loop]*100)
    #Extract the flip angle of this loop (degrees)
    if faInt != 0:
        try: 
            fa = multi*sliceProfile[faInt-1,:]
        except: 
            fa = multi*np.ones([noOfIsochromatsZ])*180
    else: 
        fa = multi*np.zeros([noOfIsochromatsZ])*180
    #Convert to radians
    thetaX = ((fa/360)*2*np.pi)  
    
    rotX = np.zeros([len(thetaX),3,3])
    rotY = np.zeros([len(thetaX),3,3])
    #rotation (pulse) flips spins from aligned with the z-axis to
    #aligned with the x-axis
    #Rotates around the x axis  
    for theta in range(len(thetaX)):
        rotX[theta,:,:] = np.array([[1, 0, 0], [0, np.cos(thetaX[theta]), np.sin(thetaX[theta])], \
                        [0, -np.sin(thetaX[theta]), np.cos(thetaX[theta])]])
        rotY[theta,:,:] = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])

    #Combined rotation (in this case same as rotX)
    vecMRotation = np.matmul(rotY,rotX) 

    # Updating the magnetization vector matricies
    #For tissue
    vecMArrayTissue = np.matmul(vecMRotation,vecMArrayTissue)
    #For blood 
    vecMArrayBlood = np.matmul(vecMRotation,vecMArrayBlood)

    return vecMArrayTissue, vecMArrayBlood

