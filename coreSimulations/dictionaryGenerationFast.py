# -*- coding: utf-8 -*-
"""-----------------------------------------------------------------------------
Dictionary generation for a MR fingerprint using a bloch simiulation of a 
two compartment model with a semipermeable barrier 

VARIATIONS IN: 
    - vb: percentage blood volume 
    - tb: intravascular water residence time 
    - T1t: T1 of tissue compartment
    - T1b: T1 of blood compartment 
    - B1+: B1 multiplication factor

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
import warnings
import io
import cProfile
import pstats
from line_profiler import LineProfiler
from perlin_noise import PerlinNoise
import random
import optim_TR_FA_gen as TF


#go up a folder
#os.chdir("..")

''' ----------------------SPECIFY PARAMETERS------------------------------ '''

#Definition of the function that calculates the parameters for each of the 
# multiprocessing threads
def parameterGeneration():
    """
    Generate the parameter combinations for the dictionary generation. 

    Returns:
    --------    
    allParams : list of tuples
        List of tuples containing the parameters for each dictionary entry
    """
    
    # To allow for multiple instances of the same code to be run simulataneously 
    # an instance is specified to ensure FA and TR array files are called 
    # to the correct simulation
    instance = 3
    
    ## ISOCHROMAT INFORMATION
    
    ## Specify dimentsions of total array (3D along x, y, and z)
    # noOfIsochromatsX MUST be divisible by the vb steps
    # i.e. if 1% blood volume steps are required then need noOfIsochromatsX = 100 
    #      if 0.1% blood volume steps are required, noOfIsochromatsX = 1000 etc.
    noOfIsochromatsX = 10000 #1000
    noOfIsochromatsY = 1
    noOfIsochromatsZ = 1 #number of samples in HALF of the profile
    # TR train length
    noOfRepetitions = 1000 #1000 #1000 FISP
    
    ## TISSUE PROPERTIES
    #Assign initial arrays for the tissue values 
    #Format:    array = [tissue value, blood value] 
    # Units: ms
    #t1Array = np.array([1300,1800])
    #t2Array = np.array([72,165]) #np.array([71,165]) # T2 blood value taken from average range from Powell et al. [55-275]ms
    t2StarArray = np.array([50,21])
     
    """
    ## DEFINITION OF VARIATIONS FOR SIMULATION
    # Specify the ranges and step sizes of the dictionary dimensions
    # intravascular water residence time (res) UNIT: ms
    resArray = range(200,1700,100) #range(200,1700,107) #range(200,1700,70) 
    # percentage blood volume (perc) UNIT: %
    percArray = range(10,110,10) #REMEMBER IT WILL BE DIVIDED BY 10 110
    #T1 of tissue compartment (t1t) UNIT: ms
    t1tArray = range(1000,2200,200) #range(700,1700,69) 
    #T1 of blood compartment (t1b) UNIT: ms
    t1bArray = range(1500,2100,200) #range(1540,1940,27) 
    # multiplication factor for the B1 value (multi)
    multiArray = range(80, 130, 10) #100

    # T2 of tissue compartment UNIT: ms
    t2tArray = range(38,127,15)
    # T2 of blood compartment UNIT: ms
    t2bArray = range(55, 385, 110)
    if t2tArray[-1] > 112:
        t2tArray= list(t2tArray)
        t2tArray[-1] = 112
    """
    """SPGRE
    t2tArray = [np.NaN]
    # T2 of blood compartment UNIT: ms
    t2bArray = [np.NaN]
    """

    """
    if percArray[-1] > 100:
        percArray= list(percArray)
        percArray[-1] = 100
    """
    
    #Testing limits
    resArray = [200] #range(200,1700,107) #range(200,1700,70) 
    # percentage blood volume (perc) UNIT: %
    percArray = [0]#[90] #REMEMBER IT WILL BE DIVIDED BY 10 110
    #T1 of tissue compartment (t1t) UNIT: ms
    t1tArray = [700] #[1400] #range(700,1700,69) 
    #T1 of blood compartment (t1b) UNIT: ms
    t1bArray = [700]#[1500] #range(1540,1940,27) 
    # multiplication factor for the B1 value (multi)
    multiArray = [100] #100
    # T2 of tissue compartment UNIT: ms
    t2tArray = [85]#[68]
    # T2 of blood compartment UNIT: ms
    t2bArray = [85] #[165]
    


    # Search space
    print('Parameter space:')
    print(f'Residence times: {list(resArray)}')
    print(f'Blood Volume Percentage: {list(percArray)}')
    print(f'T1t: {list(t1tArray)}')
    print(f'Tbt: {list(t1bArray)}')
    print(f'B1+: {list(multiArray)}')
    print(f'T2t: {list(t2tArray)}')
    print(f'T2b: {list(t2bArray)}')



    ## NOISE INFORMATION 
    # number of noise levels
    # for dictionary generation for comparison to experimental data set to one 
    noise = 1
    #The dictionary folder identifier
    #In folder will show as "DictionaryXXX" 
    #This folder needs to already exist or code will not run 

    dictionaryId  = 'Discard'

    ## SHAPE OF VARIATIONS
    # TODO: add extra shape variations documentation
   # Variations of the flip angle and repetition time have constrained shapes 
    # dictated by 5 parameters (a-e). They are also dicated by an overall shape 
    # specified here  
    # For flip angle [degrees]: 
    #       random: random variation in FA between two values: 0 and 4*a
    #       sin: absolute values of a sinusoidal variation with even peaks 
    #            half the height of the odd peaks. maximum height = 4*a 
    #            width of half peak = pi*b 
    #       gaps: same as sinusoidal but with user specifed sections of zero FA 
    #             without editing gaps will be after every 250 FAs (can be edited below)
    caseFA = 'FISPorig' #'sin' #'random'  #'gaps' 'FISP' 'FISPorig'
    b1sens= False # changes the shape of the FA train; instead of Jiang et al. variation, it becomes the variation from doi: 10.1002/mrm.26009

    # For repetition time [ms]: 
    #       random: random variation in FA between two values: d and e
    #       sin: sinusoidal variation with min TR = d, max TR = 2*e+d, period = 2*pi*c
    caseTR = 'perlin' # #'sin' #'random' 'perlin' for original FISP use last
    
    #states = [8.77345817, 147.49943504, 398.3546085, 5.81103981, 17.98150172]
    #a= states[0]; b = states[1]; c = states[2];  d = states[3]; e = states[4]
    #states = [8.90191785e+01, 8.50678055e+01, 8.88994604e+01, 8.98932576e+01,1.44671580e-03, 6.56840603e+01, 1.97728279e+02] # #ISMRM
    # B1 OPTIM UPGRADE states = [8.77080631e+01, 8.86049847e+01, 8.83192444e+01, 8.79182172e+01, 8.42228904e+01, 8.87247902e+01, 8.92251865e+01, 8.79974962e+01, 8.85240518e+01, 8.02671601e-04, 8.72068319e+01, 1.98115750e+02]
    states = [8.85357690e+01, 8.62215988e+01, 8.43992634e+01, 8.96210295e+01, 8.57002339e+01, 8.80460203e+01, 8.39164128e+01, 8.50884461e+01, 8.00508468e+01, 7.77892592e+01, 8.55644599e-04, 9.49214848e+01, 1.99304875e+02] #OPTIM no B1 sensitivity
    #If you want gaps in the flip angles, specify width here 
    if caseFA == 'gaps':
        gapLength = 80

    if caseFA == 'tester':
        faArray = [90]

    CSFnullswitch = True

    a = 13; b = 40; c =181; d = 100; e = 45 #SPGRE

    ''' ------------------------PARAMETER VARIATIONS-------------------------- '''
    
    ##  DEFINING FA ARRAY
    
    ## FIX ME
    inv = 1

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
        if inv == 1:
            #faArray[0] = 180
            faArray = np.insert(faArray,0,180)

    if caseFA == 'FISPorig':
        # From https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.25559
        # actual original fisp paper values
        faArray = np.genfromtxt('fa_jiang', delimiter=',', dtype=float) 
        if inv == 1:
            #faArray[0] = 180
            faArray = np.insert(faArray,0,180)


    if caseFA == 'FISP':
        """
        # Set the number of points
        Nrf = 200
        noOfRepetitions = 1000
        cycles = noOfRepetitions/Nrf
        faArray = []
        min_angle = 5
        #ßmaxFA = random.sample(range(5,90),int(cycles)) # random sample of flip angles
        #maxFA = [a-min_angle,b-min_angle,c-min_angle,d-min_angle,e-min_angle] # values from the paper
        maxFA = [88,88,88,88,88,88,88,88,88,88] # values from the paper
        for i in range(int(cycles)):
            # Current random flip angle
            maxFA_i = maxFA[i]
            # Iterate through the segment
            for j in range(1,Nrf):
                # Calculate the flip angle
                flipAngle = np.sin(j*np.pi/Nrf)*maxFA_i
                # Append the flip angle to the array
                faArray.append(flipAngle+min_angle)
            faArray += [0,0,0,0,0,0,0,0,0,0]
        faArray = faArray[:noOfRepetitions]

        if CSFnullswitch == True:
            faArray[0] = 180
        """
        peaks = np.array(states[:-3]) # the final three elements in the state array relate to TR, so go up to there

        faArray = TF.FISP_FA(peaks, noOfRepetitions, instance, invSwitch=True, save = False, b1Sensitivity=b1sens)

        

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

    elif caseFA =='cao':
        faArray = np.load('fa_cao.npy') 

    elif caseFA=='test':
        faArray = np.ones((noOfRepetitions,))*50

    elif caseFA=='automeris':
        faArray = np.genfromtxt('FA_FISP.csv')[:,1]
        print(faArray.shape)
        
     

    #Save array for calling in the main function later
    #np.save('./coreSimulations/functions/holdArrays/faArray_'  + str(instance) + '.npy', faArray)
    np.save('./functions/holdArrays/faArray_'  + str(instance) + '.npy', faArray)

    ##  DEFINING TR ARRAY
  
    if caseTR == 'sin':
        """
        #Generate linearly spaced array between 0 and the number repetitions 
        xRange = np.linspace(0,noOfRepetitions,num=noOfRepetitions)
        #Calculate the sinusoidal array 
        trArray = e*np.sin(xRange/c)+(d+e) #FOR SIN FA SPGRE
        if CSFnullswitch == True:
            trArray[0] = 2909, look at optim_TR_FA_gen for TI
        """
        """
        # For SPGRE
        #Generate linearly spaced array between 0 and the number repetitions 
        xRange = np.linspace(0,noOfRepetitions,num=noOfRepetitions)
        #Calculate the sinusoidal array 
        trArray = e*np.sin(xRange/c)+(d+e)
        if inv == True:
            #trArray = np.insert(trArray,0,40)# empirical value to match the original FISP paper. with no CSF nulling
            T1CSF = 4658.3 # mean value from Bojorquez et al. MRI, 2017
            trArray = np.insert(trArray,0,T1CSF*np.log(2)) # there is no pulse before the TI, and no repetitive TI pulse so use EQ 14.36a. from Bernstein book
        """
        # For FISP
        trArray = TF.sinusoidal_TR(TRmax=states[-1], TRmin=states[-2], freq=states[-3], N = noOfRepetitions, instance= instance, CSFNullSwitch = CSFnullswitch, save = False)

    elif caseTR == 'random':
        #Generate a uniform random array between for the number of repetitions
        trArray = np.random.uniform(d,e,[noOfRepetitions])
   
    elif caseTR =='perlin':
        """
        perlin = PerlinNoise(octaves=3, seed=8)
        trArray = np.array([perlin(i/noOfRepetitions) for i in range(noOfRepetitions)])
        # scale TR to our range (FIX SO THIS ISN'T HARDCODED)
        min_TR = 11.5
        max_TR = 14.5
        trArray = (trArray-min(trArray))/(max(trArray)-min(trArray))*(max_TR-min_TR)+min_TR
        """
        # https://onlinelibrary.wiley.com/doi/10.1002/mrm.25559 Perlin Noise
        trArray = np.genfromtxt('tr_jiang', delimiter=',', dtype=float) 
        if inv == 1:
            trArray = np.insert(trArray,0,1) #40
    
    elif caseTR =='cao':
        trArray = np.load('tr_cao.npy') 

    elif caseTR=='tester':
        trArray = np.ones((noOfRepetitions,))*40
        


    #Save array for calling in the main function later
    np.save('./functions/holdArrays/trArray_' + str(instance) + '.npy', trArray)

    fullSampling = False # to cover the whole parameter space
    #Get all combinations of arrays (parameters for each dictionary entry)
    #In format of list of tuples
    params = list(itertools.product(t1tArray, t1bArray, resArray, percArray, multiArray, t2tArray, t2bArray))
    
    if fullSampling == True:
        params = np.array(params)
    
    elif fullSampling == False:
        # find mean T1 tissue and T1 blood
        t1tmean = np.mean(t1tArray)
        t1bmean = np.mean(t1bArray)
        t2tmean = np.mean(t2tArray)
        t2bmean = np.mean(t2bArray)

        # remove params from the search space:
        # if T1 tissue is above the mean, T2 tissue below the mean should be removed from list of params in itertools
        filtered_combinations = [
            (t1t, t1b, res, perc, multi, t2t, t2b) for t1t, t1b, res, perc, multi, t2t, t2b, in params
            if not (t1t > t1tmean and t2t < t2tmean) and not(t1b > t1bmean and t2b < t2bmean) and not (t2t > t2tmean and t1t < t1tmean) and not (t2b > t2bmean and t1b < t1bmean) 
        ]

        params =  np.array(filtered_combinations)
        print(f'Calculating {len(params)} parameter combinations.')

    #Generate a list of the remaining parameters than need to be passed into
    #the main function 
    otherParams = list([t2StarArray, noOfIsochromatsX, noOfIsochromatsY,
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
    """
    Function to generate a dictionary entry for a given set of parameters.

    Parameters:
    ----------- 
    paramArray : list
        List of parameters for one dictionary entry
    """
    
    sys.path.insert(0, "./functions/")
    from DictionaryGeneratorFast import DictionaryGeneratorFast
    
    #Is there an inversion pulse, THIS NEEDS REMOVING IT DOES NOTHING
    invSwitch = True
    # Is slice profile accounted for
    sliceProfileSwitch = 0
    #Null CSF using inversion
    CSFnullswitch = False
    # Number of noise samples generated 
    # Set to one for dictionary gneeration 
    samples = 1
    
    parameters = tuple(paramArray)

    t1Array = np.array([parameters[0],parameters[1]])
    #These parameters are: 
 # t1Array, t2Array, t2StarArray, noOfIsochromatsX,
 #              noOfIsochromatsY, noOfIsochromatsZ, noOfRepetitions, noise, perc, res,
 #              multi, inv, CSFnullswitch, sliceProfileSwitch, samples, dictionaryId, sequence, instance
    t2Array = np.array([parameters[5],parameters[6]])
    #Initialise the dictionary generator
    dictionaryGenerator =  DictionaryGeneratorFast(t1Array, t2Array,
            parameters[7], parameters[8], parameters[9],parameters[14],
            parameters[10], parameters[11], parameters[3]/10, parameters[2],
            parameters[4]/100,invSwitch, CSFnullswitch, sliceProfileSwitch, samples, parameters[12], 'FISP', parameters[13])
    
    profile = False # Set to True to profile the function to test for bottle necks (default is False)

    if profile == True:
        # Create a Profile object
        pr = cProfile.Profile()
        # Enable profiling
        pr.enable()
    
        # Run the dictionary gener ation
        dictionaryGenerator.MRFSGRE()
        
        # Disable profiling
        pr.disable()
        # Save the profile results
        s = io.StringIO()
        # Create a Stats object from the Profile object, and write the results to the StringIO object
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        # Print the profiling results
        ps.print_stats()
        print(s.getvalue())

    else:
        """
        # Run the dictionary generation
        lp = LineProfiler()
        lp.add_function(dictionaryGenerator.MRFSGRE)
        wrapper = lp(dictionaryGenerator.MRFSGRE)
        wrapper()   
        lp.dump_stats("profile_results.txt")
        lp.print_stats()
        """
        dictionaryGenerator.MRFSGRE()
    # Return the result
    return None


'''-------------------------MAIN DICTIONARY LOOPS---------------------------'''


if __name__ == '__main__':
    
    #Multiprocessing requires all moduels used within the threads to be defined
    #within __main__
    #I think this is a safety feature
    #os.chdir("./functions/")
    #sys.path.insert(0, "./functions/")
    import time
    import itertools
    import multiprocessing as mp


    print('Beginning dictionary generation...')

    #For multiprocessing use the number of available cpus  
    #Currently set to perform differently on my Mac ('Darwin') system vs the cluster
    if platform.system() == "Darwin":
        #If on local computer can use all CPUs
        pool = mp.Pool(12) #12
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