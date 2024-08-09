import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt  # to plot
#import math
import os
import numpy.matlib as mat 
import sys
import optim_TR_FA_gen as TRFA

sys.path.insert(0, "./functions/")
from DictionaryGeneratorFast import DictionaryGeneratorFast

def func(a_max, w_a, freq, TRmin, TRmax):
        # change this code so the input is a parameter array, put if statements for each generation of parameters to keep it general
    # Initial guess
        noOfRepetitions = 2000
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
        instance = 25
        noise = 1 # no noise
        multi = 100
        invSwitch = 1
        CSFnullswitch = True
        sliceProfileSwitch = True
        sequence = 'FISP'
        
        trArray = TRFA.sinusoidal_TR(TRmax, TRmin, freq, noOfRepetitions, instance, save = True)
        faArray = TRFA.sinusoidal_FA(a_max, noOfRepetitions, w_a, instance, save = True)
        readTRFA = False
        

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
        #return func(permutation[0], permutation[1], permutation[2], permutation[3], permutation[4])
        return func(permutation[0], permutation[1], permutation[2], permutation[3], permutation[4])