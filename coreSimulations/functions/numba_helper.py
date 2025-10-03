# Numba helper functions
import numpy as np
from numba import njit, jit, types
from numba.extending import overload
from numba.core.errors import TypingError


@njit
def sincos(x):
    return np.sin(x), np.cos(x)

"""Numpy implementations that need to be copied for jit"""
@jit(nopython=True, fastmath = True,  cache=True)
def jit_repeat(arr, repeats, axis = None):
    """Replicates np repeat for a 3D array for axis = 2"""
    rows, cols,_ = arr.shape
    repeated_arr = np.empty((rows, cols, repeats), dtype=arr.dtype)
    for rep in range(repeats):
        repeated_arr[:,:,rep] =  arr[:,:,0] 
    return repeated_arr

@jit(nopython=True, fastmath = True,  cache=True)
def jit_matmul(M, v):
    """
    Equivalent to einsum('...ij,...j->...i'). Batched matrix mult
    matrices (ndarray): Shape (dim1, dim2, dim3, 3, 3)
    vectors (ndarray): Shape (dim1, dim2, dim3, 3)
    Return shape (dim1, dim2, dim3, 3)
    """
    dim1, dim2,dim3 = v.shape[:3]

    multiplied = np.empty_like(v)
    for i in range(dim1):
         for j in range(dim2):
              for k in range(dim3):
                    multiplied[i, j, k] = M[i, j, k] @ v[i, j, k]

    return multiplied 



@jit(nopython=True, fastmath = True, cache=True)
def rotation_calculations_numba(gradientX, positionArrayX, gradientY, positionArrayY, noOfIsochromatsZ, deltaT):
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

        # expand gradient matrix
        exp_gradient = np.expand_dims((42.6) * gradientMatrix, axis=2)

        # Gyromagnetic ratio for proton 42.6 MHz/T 42.576
        omegaArray = jit_repeat(exp_gradient, noOfIsochromatsZ, axis=2)
        

        #for the precessions generate an array storing the 3x3 rotation matrix 
        #for each isochromat
        precession = np.zeros((omegaArray.shape[0], omegaArray.shape[1], omegaArray.shape[2], 3,3), dtype=np.float64)
        precession[:,:,:,2,2] = 1
   
        # compute the trigonometric functions for the rotation matrices
        sin_omega_deltaT, cos_omega_deltaT = sincos(omegaArray*deltaT)

        precession[:,:,:,0,0] = cos_omega_deltaT
        precession[:,:,:,0,1] = -sin_omega_deltaT
        precession[:,:,:,1,0] = sin_omega_deltaT
        precession[:,:,:,1,1] = cos_omega_deltaT
   
        return precession


"""Main functions"""
@jit(nopython=True, fastmath = True, cache=True)
def applied_precession_numba(gradientDuration, deltaT, totalTime, precessionTissue, vecMArrayTissue, exp_delta_t_t2_star_tissue, 
                             one_minus_exp_delta_t_t1_tissue, exp_delta_t_t1_tissue, precessionBlood, vecMArrayBlood, exp_delta_t_t2_star_blood,
                             one_minus_exp_delta_t_t1_blood, exp_delta_t_t1_blood, signalDivide, signal_flag, signal):
     # Performs the time loop for the applied precession, vecMArrayTissue and vecMArrayBlood should be squeezed before input
     
    
    num_tissue_isos = vecMArrayTissue.shape[0]
     #For each time step
    for tStep in range(int(gradientDuration/deltaT)):

        #Update time passed
        totalTime += deltaT

        #Multiply by the precession rotation matrix (incremental for each deltaT)
        vecMThold = jit_matmul(precessionTissue,vecMArrayTissue)
        
        # The magnitude change due to relaxation is then applied to each
        # coordinate
        vecMThold[...,0] *= exp_delta_t_t2_star_tissue
        vecMThold[...,1] *= exp_delta_t_t2_star_tissue
        vecMThold[...,2] = one_minus_exp_delta_t_t1_tissue + vecMThold[...,2]*exp_delta_t_t1_tissue
        #The stored array is then updated
        vecMArrayTissue = vecMThold
        
        # do the same for blood
        #Multiply by the precession rotation matrix (incremental for each deltaT)
        vecMBhold = jit_matmul(precessionBlood, vecMArrayBlood)
        
        # The magnitude change due to relaxation is then applied to each
        # coordinate
        vecMBhold[...,0] *= exp_delta_t_t2_star_blood
        vecMBhold[...,1] *= exp_delta_t_t2_star_blood
        vecMBhold[...,2] = one_minus_exp_delta_t_t1_blood + vecMBhold[...,2]*exp_delta_t_t1_blood
        #The stored array is then updated
        vecMArrayBlood = vecMBhold
        
        #If the total time that has passed corresponds to the time at which
        # there is an echo peak:
        if signal_flag == True: 
            time_step_int = int(totalTime/deltaT)
            for i in range(len(signalDivide)):
                if time_step_int == signalDivide[i]:
                    ind = i
                    # need to squeeze out extra dimension in 1st axis
                    d0, d2, d3 = vecMArrayTissue.shape[0], vecMArrayTissue.shape[2], vecMArrayTissue.shape[3]
                    signal[:num_tissue_isos, 0, :, :, ind] = vecMArrayTissue.reshape((d0, d2, d3))
                    d0, d2, d3 = vecMArrayBlood.shape[0], vecMArrayBlood.shape[2], vecMArrayBlood.shape[3] 
                    signal[num_tissue_isos:, 0, :, :, ind] = vecMArrayBlood.reshape((d0, d2, d3))
                    
                    break # Found the match

    # add squeezed dimensions back   
    vecMArrayTissue = np.expand_dims(vecMArrayTissue, axis=4)
    vecMArrayBlood = np.expand_dims(vecMArrayBlood, axis=4)

    return signal, totalTime, vecMArrayTissue, vecMArrayBlood