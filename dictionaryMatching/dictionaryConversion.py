#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""------------------------------------------------------------------------
Compress the dictionary into two files 
- Scale dictionary to have 1000 grey points 

Author: Emma Thomson
Year: 2022
Institution: Centre for Medical Image Computing: University College London
Email: e.thomson.19@ucl.ac.uk
------------------------------------------------------------------------"""

''' -----------------------------PACKAGES--------------------------------- '''

import numpy as np
import os
import glob

import time
print(os.getcwd())

t0 = time.time()
import itertools
import warnings
warnings.filterwarnings("ignore")

#go up a folder
#os.chdir("..")


''' -----------------------------INPUTS--------------------------------- '''
acqlen = 1000

# Dictionary folder
dictfolder = 'FISP_WEX_ISMRM'

#Type of normalisation performed 
#Set to L2-norm (standard)
norm_type = 2 #1 #2

''' --------------------------READ IN SIMS------------------------------ '''

print("Starting Dictionary Read in:  " + str(time.strftime('%X %x %Z')))

#Number of entries in dictionary (to set array sizes)
dictPath = '../dictionaries/Dictionary' + dictfolder + '/'

##### comment out if not filtering dictionary#################
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
if percArray[-1] > 100:
    percArray= list(percArray)
    percArray[-1] = 100
params = list(itertools.product(t1tArray, t1bArray, t2tArray, t2bArray, resArray, percArray, multiArray))
# find mean T1 tissue and T1 blood
t1tmean = np.mean(t1tArray)
t1bmean = np.mean(t1bArray)
t2tmean = np.mean(t2tArray)
t2bmean = np.mean(t2bArray)

# remove params from the search space:
# if T1 tissue is above the mean, T2 tissue below the mean should be removed from list of params in itertools
filtered_combinations = [
    (t1t, t1b, t2t, t2b, res, perc, multi) for t1t, t1b, t2t, t2b, res, perc, multi in params
    if not (t1t > t1tmean and t2t < t2tmean) and not(t1b > t1bmean and t2b < t2bmean) and not (t2t > t2tmean and t1t < t1tmean) and not (t2b > t2bmean and t1b < t1bmean) 
]
##########################

#no_entries = np.size( [f for f in os.listdir(dictPath) if f.endswith('.npy')])
no_entries = len(filtered_combinations)


#Open empy arrays for dictionary signals
array = np.zeros((acqlen,no_entries))
mean_sims = np.zeros([no_entries])
files = []

count = 0
sample = [0,0]
rr = '*.npy' #rr = '*1.0_1.npy'
convert_to_mag = True # whether to convert to a magnitude signal from complex
#Loading all dictionary signals into array    
#for filename in glob.glob(os.path.join(str(dictPath + rr))):
for param in filtered_combinations:
    filename = f'../dictionaries/DictionaryFISP_WEX_ISMRM/echo_{param[0]}_{param[1]}_{param[2]}_{param[3]}_{param[4]}_{param[5]/10}_{param[6]/100}_1.npy'
    print(f"Processing file: {count+1}")
    with open(os.path.join(filename), 'r') as f:
        #No water exchange variation considered
        #filesplit = filename.split('_')
        #if filesplit[3] == '300':
        if convert_to_mag == False:
            hold = np.load(filename)
        else:
            hold_complex = np.load(filename)
            hold = np.sqrt((hold_complex[:,0])**2 + (hold_complex[:,1])**2) # take the magnitude of the data
        
        #Divided by the number of isochromats to give a fractional value
        try:
            # Save dictionary signals into array
            if convert_to_mag == False:
                array[:,count] = np.squeeze(hold[0:acqlen,0]) 
            else:
                array[:,count] = hold[0:acqlen]
            #Save parameter values  (Look up table)
            strsplit = filename.split('_')  
            files.append([float(strsplit[3]),float(strsplit[4]), # change this depending on the file structure (print the filename to see what it looks like)
                            float(strsplit[5]),float(strsplit[6]),
                            float(strsplit[7]), float(strsplit[8]), float(strsplit[9])])
            count += 1
        except: 
            #Skip if it doesnt work 
            fff = 2
            print("Likely error in file path")
    



print("Starting Normalisation:  " + str(time.strftime('%X %x %Z'))) # between 0 and 1000

for ii in range(0,count):
    array[:,ii] = (array[:,ii]- np.min(array[:,ii]))/(np.max(array[:,ii])-np.min(array[:,ii]))*1000

print("Starting Saving:  " + str(time.strftime('%X %x %Z')))
array = np.squeeze(array.astype(int)).T
np.savetxt(dictPath + "dictionary.txt", array, fmt="%i", delimiter=",", newline="\n")

#Turn the look up table into an array
lookup = np.array(files)
np.savetxt(dictPath + "lookupTable.txt", lookup, delimiter=",", newline="\n")

#Stop timer and print                                                    
t1 = time.time()
total = t1-t0
print("Total Time:    " +  str(total) + " seconds")  


