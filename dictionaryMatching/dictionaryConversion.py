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

t0 = time.time()

import warnings
warnings.filterwarnings("ignore")

#go up a folder
#os.chdir("..")


''' -----------------------------INPUTS--------------------------------- '''
acqlen = 2000

# Dictionary folder
dictfolder = 'Test'

#Type of normalisation performed 
#Set to L2-norm (standard)
norm_type = 2 #1 #2

''' --------------------------READ IN SIMS------------------------------ '''

print("Starting Dictionary Read in:  " + str(time.strftime('%X %x %Z')))

#Number of entries in dictionary (to set array sizes)
dictPath = './MRFSGRE_BBB/Dictionaries/Dictionary' + dictfolder + '/'
no_entries = np.size( [f for f in os.listdir(dictPath) if f.endswith('.npy')])

#Open empy arrays for dictionary signals
array = np.zeros((acqlen,no_entries))
mean_sims = np.zeros([no_entries])
files = []

count = 0
sample = [0,0]
rr = '*.npy' #rr = '*1.0_1.npy'
#Loading all dictionary signals into array    
for filename in glob.glob(os.path.join(str(dictPath + rr))):
    print(f"Processing file: {count+1}")
    
    with open(os.path.join(filename), 'r') as f:
        #No water exchange variation considered
        #filesplit = filename.split('_')
        #if filesplit[3] == '300':
        hold = np.load(filename)
        #Divided by the number of isochromats to give a fractional value
        try:
            # Save dictionary signals into array
            array[:,count] = np.squeeze(hold[0:acqlen,0]) 
            #Save parameter values  (Look up table)
            strsplit = filename.split('_')   
            files.append([float(strsplit[2]),float(strsplit[3]), # change this depending on the file structure (print the filename to see what it looks like)
                            float(strsplit[4]),float(strsplit[5]),
                            float(strsplit[6])])
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


