import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt  # to plot
#import math
import os
import numpy.matlib as mat 
import sys
import optim_TR_FA_gen as TRFA



from DictionaryGeneratorFast import DictionaryGeneratorFast

""" Cost function for signal optimisation"""

def func(permutation):

        if len(permutation)== 5: # this size is for a trType == 'sin' and faType == 'sin', i.e. Emma's variations
                a_max=permutation[0]; w_a=permutation[1]; freq=permutation[2]; TRmin=permutation[3]; TRmax=permutation[4]

        else: # this size is for a trType == 'sin' and faType == 'FISP'
                FISP_FA = True
                num_peaks = len(permutation) - 3 # final parameters are fixed, the number of peaks changes
                peaks= np.array(permutation[:num_peaks]); freq=permutation[-3]; TRmin=permutation[-2]; TRmax=permutation[-1]

        # change this code so the input is a parameter array, put if statements for each generation of parameters to keep it general
        # Initial guess
        if FISP_FA == True:
                noOfRepetitions = 2000 # for now this is limited to TR = 1000 or 2000 (i.e. num_peaks should be either 5 or 10)
        else:
                noOfRepetitions = 1000
        
        noOfIsochromatsX = 1000
        noOfIsochromatsY = 1
        noOfIsochromatsZ = 10


        # Values of GM T1 tissue/blood, Blood volume (vb), GM T2 tissue from Zhao et al. 2007 
        # T2 blood value taken from average range from Powell et al. [55-275]ms

        # Format = [tissue, blood]
        t1Array = np.array([1122,1627])
        t2Array = np.array([71,165])
        t2StarArray = np.array([50,21])
        res_min = 200
        res_max = 1600
        vb = 50

        # Sequence parameters
        samples = 1
        dictionaryId = 'Discard'
        instance = 62
        noise = 1 # no noise
        multi = 100
        invSwitch = 1
        CSFnullswitch = False # we only rely on invSwitch now
        sliceProfileSwitch = True
        sequence = 'FISP'
        trType = 'sin'
        faType = 'FISP'
        b1switch = False # if we want the B1 consideration
        
        if trType == 'sin' and faType == 'sin':
                trArray = TRFA.sinusoidal_TR(TRmax, TRmin, freq, noOfRepetitions, instance, save = True)
                faArray = TRFA.sinusoidal_FA(a_max, noOfRepetitions, w_a, instance, save = False)
      
        elif trType == 'sin' and faType == 'FISP':
                trArray = TRFA.sinusoidal_TR(TRmax, TRmin, freq, noOfRepetitions, instance, save = True)
                faArray = TRFA.FISP_FA(peaks, noOfRepetitions, instance, save = True, b1Sensitivity = b1switch)
        
        
        readTRFA = False # flag for whether to read TR/FA from a file. This can intefere with parallelism
        
        dictionaryGeneratorMin = DictionaryGeneratorFast(t1Array, t2Array, t2StarArray, noOfIsochromatsX,\
                noOfIsochromatsY, noOfIsochromatsZ, noOfRepetitions, noise, vb/10, res_min,\
                multi/100, invSwitch, CSFnullswitch, sliceProfileSwitch, samples, dictionaryId, sequence, instance, readTRFA, trArray, faArray)
        
        dictionaryGeneratorMax = DictionaryGeneratorFast(t1Array, t2Array, t2StarArray, noOfIsochromatsX,\
                noOfIsochromatsY, noOfIsochromatsZ, noOfRepetitions, noise, vb/10, res_max,\
                multi/100, invSwitch, CSFnullswitch, sliceProfileSwitch, samples, dictionaryId, sequence, instance, readTRFA, trArray, faArray)
        
        
        signalMin = dictionaryGeneratorMin.MRFSGRE()[:,0]
        signalMax = dictionaryGeneratorMax.MRFSGRE()[:,0]

        return -np.sqrt(np.mean((signalMin - signalMax) ** 2)) # - rmse

def costFunction(permutation):
        """ Cost of x = f(x)."""
        return func(permutation)
