# -*- coding: utf-8 -*-
"""-----------------------------------------------------------------------------
Dictionary generation for a MR fingerprint using a bloch simiulation of a 
two compartment model with a semipermeable barrier 

VARIATIONS IN: 
    - vb: percentage blood volume 
    - tb: intravascular water residence time 
    - T1t: T1 of tissue compartment
    - T1b: T1 of blood compartment 

TO DO: 
    - Need to add additional tissue compartment to elevate partial volume effects 
    

Author: Emma Thomson
Year: 2022
Institution: Centre for Medical Image Computing: University College London
Email: e.thomson.19@ucl.ac.uk
----------------------------------------------------------------------------"""

''' -----------------------------PACKAGES--------------------------------- '''
## IMPORT DEPENDANCIES
import numpy as np
import sys
import os
import platform
if platform.system() == "Darwin":
    sys.path.insert(0, "/Users/emmathomson/Dropbox/Coding/BBB_MRFSGRE/coreSimulations/functions/")
else: 
    sys.path.insert(0, "/home/ethomson/Coding/Fingerprinting/coreSimulations/functions/")
from blochSimulation import MRFSGRE
'''
os.chdir("/Users/emmathomson/Dropbox/Coding/Python-Codes/noise/")
from noise import pnoise1
os.chdir("/Users/emmathomson/Dropbox/Coding/BBB_MRFSGRE/coreSimulations/")

def perlin(span, tr_min, tr_max):

    if len(sys.argv) > 1:
    	octaves = int(sys.argv[1])
    else:
    	octaves = 1
        
    base = 0
    #min = max = 0
    
    x_arr  = []
    y_arr = []
    
    points = 2000
    r = range(points)
    
    for i in r:
        x = float(i) * span / points - 0.5 * span
        y = pnoise1(x + base, octaves)
        x_arr.append([x])
        y_arr.append([y])
        
    #normalize 
    norm_y = (y_arr - np.min(y_arr)) / (np.max(y_arr) - np.min(y_arr))
    TR_array = np.round((norm_y*(tr_max-tr_min))+tr_min,1)
    #TR_array = y_arr
    
    return TR_array
'''
''' ----------------------SPECIFY PARAMETERS------------------------------ '''

#Definition of the function that calculates the parameters for each of the 
# multiprocessing threads
def parameterGeneration():
    # To allow for multiple instances of the same code to be run simulataneously 
    # an instance is specified to ensure FA and TR array files are called 
    # to the correct simulation
    instance = 1
    
    ## ISOCHROMAT INFORMATION
    
    ## Specify dimentsions of total array (2D along x and y)
    # noOfIsochromatsX MUST be divisible by the vb steps
    # i.e. if 1% blood volume steps are required then need noOfIsochromatsX = 100 
    #      if 0.1% blood volume steps are required, noOfIsochromatsX = 1000 etc.
    noOfIsochromatsX = 1000
    noOfIsochromatsY = 1
    noOfIsochromatsZ = 10
    # TR train length
    ## FIX ME
    noOfRepetitions = 2000
    
    ## TISSUE PROPERTIES
    #Assign initial arrays for the tissue values 
    #Format:    array = [tissue value, blood value] 
    # Units: ms
    #t1Array = np.array([1300,1800])
    t2Array = np.array([90,63])
    t2StarArray = np.array([50,21])
     
     ## FIX ME
    inv = True
    ## DEFINITION OF VARIATIONS
    
    # Specify the ranges and step sizes of the dictionary dimensions
    # intravascular water residence time (res) UNIT: ms

    resArray =  range(200,1700,107) #range(200,1700,70) 
    # percentage blood volume (perc) UNIT: %
    percArray = range(10,110,7) #REMEMBER IT WILL BE DIVIDED BY 10 
    #T1 of tissue compartment (t1t) UNIT: ms
    t1tArray = range(700,1700,69) 
    #T1 of blood compartment (t1b) UNIT: ms
    t1bArray = range(1540,1940,27) 
    # multiplication factor for the B1 value (multi)
    multiArray = range(70, 120, 3) 
    #efficiency of the ir pulse 

    ## NOISE INFORMATION 
    # number of noise levels\\\\\\\\\\\\\\\\\\\\\
    #  for dictionary generation for comparison to experimental data set to one 
    noise = 1
    # number of times each noise level is samples
    # for dictionary generation for comparison to experimental data set to one 
    #noiseSamples =  1#50
    
    #The dictionary folder identifier
    #In folder will show as "DictionaryXXX" 
    #This folder needs to already exist or code will not run 
    
    #TO NOTE: SAMPLES HAS BEEN SET TO 50 

    dictionaryId  = 'IsosampleLarge'

    
    ## SHAPE OF VARIATIONS
    
   # Variations of the flip angle and repetition time have constrained shapes 
    # dictated by 5 parameters (a- e). They are also dicated by an overall shape 
    # specified here  
    # For flip angle [degrees]: 
    #       random: random variation in FA between two values: 0 and 4*a
    #       sin: absolute values of a sinusoidal variation with even peaks 
    #            half the height of the odd peaks. maximum height = 4*a 
    #            width of half peak = pi*b 
    #       gaps: same as sinusoidal but with user specifed sections of zero FA 
    #             without editing gaps will be after every 250 FAs (can be edited below)
    caseFA = 'sin' #'random'  #'gaps'
    a = 13; b = 40
    # For repetition time [ms]: 
    #       random: random variation in FA between two values: d and e
    #       sin: sinusoidal variation with min TR = d, max TR = 2*e+d, period = 2*pi*c
    caseTR = 'sin' # #'sin' #'random' 
    c =181; d = 100; e = 45
    
    #If you want gaps in the flip angles, specify width here 
    if caseFA == 'gaps':
        gapLength = 50

    ''' ------------------------PARAMETER VARIATIONS-------------------------- '''
    
    ##  DEFINING FA ARRAY
    
    if caseFA == 'sin':
        #Generate linearly spaced array between 0 and the number repetitions 
        xRange = np.array(range(noOfRepetitions))
        #Calculate the sinusoidal array 
        #Sometimes a random variation is added to the flip angle to reduced 
        #effects of matching errors with inhomogeneous B1 fields 
        #Hence there is a random element that can be added (set currently to 0)
        faArray = np.squeeze(a*(np.abs(3*np.sin(xRange/b)+(np.sin(xRange/b)**2))) \
            +  np.random.uniform(0,0,[1,noOfRepetitions]))
        # There may be a loss of SNR when the first flip angle is v low (e.g. 0.008)
        # so change the first flip angle to be a little higher
        if inv == True:
            faArray[0] = 180
    elif caseFA == 'gaps':
        #As above: 
        xRange = np.array(range(noOfRepetitions))
        faArray = a*(np.abs(3*np.sin(xRange/b)+(np.sin(xRange/b)**2))) \
         +  np.random.uniform(0,0,[1,noOfRepetitions]) 
        faArray[:,0] = 3
        #Add gaps to flip angle variation
        #Just inserts zeros into array does not change underlying sinusoidal shape
        noOfGaps = noOfRepetitions/250
        #Add gaps every 250 repetitions
        for ii in range(1,noOfGaps):
            faArray[250*ii:250*ii+gapLength] = 0
      
    #Save array for calling in the main function later
    if platform.system() == "Darwin":
        np.save('./holdArrays/faArray_'  + str(instance) + '.npy', faArray)
    else:
        np.save('/home/ethomson/Coding/Fingerprinting/coreSimulations/functions/holdArrays/faArray_'  + str(instance) + '.npy', faArray)
  
    ##  DEFINING TR ARRAY
    
    if caseTR == 'sin':
        #Generate linearly spaced array between 0 and the number repetitions 
        xRange = np.linspace(0,noOfRepetitions,num=noOfRepetitions)
        #Calculate the sinusoidal array 
        trArray = e*np.sin(xRange/c)+(d+e)
        if inv == True:
            trArray[0] = 2909
    elif caseTR == 'random':
        #Generate a uniform random array between for the number of repetitions
        trArray = np.random.uniform(d,e,[noOfRepetitions])
    '''   
    elif caseTR == 'perlin':
        trArray = perlin(c,d,e)
    '''
    #Save array for calling in the main function later
    
    if platform.system() == "Darwin":
        np.save('./holdArrays/trArray_' + str(instance) + '.npy', trArray)
    else: 
         np.save('/home/ethomson/Coding/Fingerprinting/coreSimulations/functions/holdArrays/trArray_' + str(instance) + '.npy', trArray)
    #Get all combinations of arrays (parameters for each dictionary entry)
    #In format of list of tuples
    params = list(itertools.product(t1tArray, t1bArray, resArray, percArray, multiArray))
    params = np.array(params)
    #Generate a list of the remaining parameters than need to be passed into
    #the main function 
    otherParams = list([t2Array, t2StarArray, noOfIsochromatsX, noOfIsochromatsY,
                        noOfRepetitions,noise, dictionaryId, instance, noOfIsochromatsZ])
    otherParams = np.tile(np.array(otherParams, dtype=object),[np.size(params,0),1])
    
    #For each tuple (dicti|onary entry) concatenate with the other parameters 
    #to be passed to generate a single tuple to be passed
    #Convert tuple to array
    allParams = list(np.append(params, otherParams, axis=1)) 
      
    return allParams

#Definition of the function that will be parallised
#Requires a single argument for parallelisation to work so previous function 
#concatenated all parameters into one list of tuples
def simulationFunction(paramArray):
    
    invSwitch = 0
    sliceProfileSwitch = 0
    samples = 1

    parameters = tuple(paramArray)
    
    t1Array = np.array([parameters[0],parameters[1]])
    MRFSGRE(t1Array, parameters[5], parameters[6],
            parameters[7], parameters[8], parameters[13],
            parameters[9], parameters[10], parameters[3]/10, parameters[2],
            parameters[4]/100,invSwitch, sliceProfileSwitch, samples, parameters[11], parameters[12])
    

'''-------------------------MAIN DICTIONARY LOOPS---------------------------'''


if __name__ == '__main__':
    
    #Multiprocessing requires all moduels used within the threads to be defined
    #within __main__
    #I think this is a safety feature
    import os
    if platform.system() == "Darwin":
        os.chdir("./functions/")
    import time
    import itertools
    import multiprocessing as mp
    #For multiprocessing use the number of available cpus  
    if platform.system() == "Darwin":
        #If on local computer can use all CPUs
        pool = mp.Pool(10)
    else:
        #If on cluster only use a few 
        pool = mp.Pool(8)
    #Start timer
    t0 = time.time()
    #Generate the parameters
    params = parameterGeneration()
    
    #Run main function in parallel 
    #Current laptop (2021 M1 Macbook Pro) will have 8 CPUs available
    try:
        pool.map(simulationFunction, params)
    finally:
        #Terminate and join the threads after parallelisation is done
        pool.terminate()
        pool.join()
        pool.close()
     
    #Stop timer and print                                                    
    t1 = time.time()
    total = t1-t0
    print(total)   
