import numpy as np
import numpy.random as rn
#import math
import os
import numpy.matlib as mat 
import sys

""" Helper functions for singal optmisation"""

""" TR variation Options"""

def sinusoidal_TR(TRmax, TRmin, freq, N, instance, CSFnullswitch = True, save = True):
    
    N_arr = np.linspace(0,N,num=N)
    #trArray = (TRmax*np.sin(N_arr/freq)+(TRmax+TRmin))
    trArray = 0.5*((TRmax-TRmin)*np.sin(N_arr*freq/2*np.pi)+(TRmax+TRmin))

    if CSFnullswitch == True:
        trArray[0] = 2909

    if save == True:
     #Save array for calling in the main function later
        np.save('./functions/holdArrays/trArray_' + str(instance) + '.npy', trArray)
    
    return trArray

def fourier_TR(A_0, A_k, B_k, T, N_ter, N):
    # Time points
    t = np.linspace(0, T, N)

    # Compute the Fourier series
    trArray = A_0 + np.zeros_like(t)
    for n in range(1, N_ter + 1):
        trArray += A_k[n-1] * np.cos(2 * np.pi * n * t / T) + B_k[n-1] * np.sin(2 * np.pi * n * t / T)
    return trArray

""" FA variation OPTIONS"""

def sinusoidal_FA(a_max, N, w_a, instance, invSwitch = True, save = True):
    N_arr = np.array(range(N))
    faArray = np.squeeze(a_max*(np.abs(3*np.sin(N_arr/w_a)+(np.sin(N_arr/w_a)**2)))) # sinusoid has a maximum value of 4, hence times by 0.25 to control the height
    
    if invSwitch == True: # if an initial inversion is desired
        faArray[0] = 180

    #Save array for calling in the main function later
    if save == True:
        np.save('./functions/holdArrays/faArray_'  + str(instance) + '.npy', faArray)
    return faArray


def FISP_FA(peaks, N, instance, invSwitch = True, save = True):

    # variation from Jiang Paper. Peaks = 1 x 5 input indicating the peaks of the variation
    
    Nrf = 200 # width of peaks
    cycles = N/Nrf
    faArray = []
    min_angle = 5 # minimum flip angle
    peaks = peaks - min_angle

    for i in range(int(cycles)):
        # Current segment maximum flip angle
        maxFA_i = peaks[i]
        # Iterate through the segment
        for j in range(1,Nrf):
            # Calculate the flip angle
            flipAngle = np.sin(j*np.pi/Nrf)*maxFA_i
            # Append the flip angle to the array
            faArray.append(flipAngle+min_angle)
        faArray += [0,0,0,0,0,0,0,0,0,0] # add recovery time between cycles
    faArray = faArray[:N]
    
    if invSwitch == True: # if an initial inversion is desired
        faArray[0] = 180

    #Save array for calling in the main function later
    if save == True:
        np.save('./functions/holdArrays/faArray_'  + str(instance) + '.npy', faArray)
    return faArray