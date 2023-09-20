#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""------------------------------------------------------------------------
Scale dictionary to have 1000 grey points 

Author: Emma Thomson
Year: 2022
Institution: Centre for Medical Image Computing: University College London
Email: e.thomson.19@ucl.ac.uk
------------------------------------------------------------------------"""


''' -----------------------------PACKAGES--------------------------------- '''

import numpy as np
import os
import glob
#import shutup
#shutup.please()

import time

t0 = time.time()

''' -----------------------------INPUTS--------------------------------- '''
acqlen = 2000

# Dictionary folder
dictfolder = 'HYDRAXXL'

#Type of normalisation performed 
#Set to L2-norm (standard)
norm_type = 2 #1 #2

''' --------------------------READ IN SIMS------------------------------ '''

print("Starting Dictionary Read in:  " + str(time.strftime('%X %x %Z')))

#Number of entries in dictionary (to set array sizes)
dictPath = '/Users/emmathomson/Desktop/Local/Dictionaries/Dictionary' + dictfolder + '/'
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
    with open(os.path.join(filename), 'r') as f:
        #No water exchange variation considered
        #filesplit = filename.split('_')
        #if filesplit[3] == '300':
            hold = np.load(filename)
            #Divided by the number of isochromats to give a fractional value
            try:
                array[:,count] = np.squeeze(hold[0:acqlen,0])
                #Save parameter values  (Look up table)
                strsplit = filename.split('_')   
                files.append([float(strsplit[1]),float(strsplit[2]),
                              float(strsplit[3]),float(strsplit[4]),
                              float(strsplit[5])])
                count += 1
            except: 
                fff = 2
        #else: 
        #    count = count 

print("Starting Normalisation:  " + str(time.strftime('%X %x %Z')))
'''
for ii in range(0,acqlen):
    array[ii,:] = (array[ii,:]/(np.linalg.norm(array[ii,:],norm_type))) 

'''
for ii in range(0,count):
    #mean_sims[ii] = np.mean(array[:,ii]) #(np.linalg.norm(array[:,ii],norm_type))
    array[:,ii] = (array[:,ii]- np.min(array[:,ii]))/(np.max(array[:,ii])-np.min(array[:,ii]))*1000
    #array[:,ii] = (array[:,ii]/(np.linalg.norm(array[:,ii],norm_type)))
    #array[:,ii] = (array[:,ii]-np.min(array[:,ii]))/(np.max(array[:,ii]) - np.min(array[:,ii])) #* mean_data
#array = array/np.linalg.norm(data[~np.isnan(data)])

#array = (array/(np.linalg.norm(array,norm_type)))
#array = (array- np.min(array))/(np.max(array)-np.min(array))*1000

print("Starting Saving:  " + str(time.strftime('%X %x %Z')))
array = np.squeeze(array.astype(int)).T
np.savetxt(dictPath + "dictionary.txt", array, fmt="%i", delimiter=",", newline="\n")

#Turn the look up table into an array
lookup = np.array(files).T
np.savetxt(dictPath + "lookupTable.txt", lookup, delimiter=",", newline="\n")

    
#Stop timer and print                                                    
t1 = time.time()
total = t1-t0
print("Total Time:    " +  str(total) + " seconds")  


'''
FOR LOADING IN THE DICTIONARY
with open(dictPath + "dictionary.txt") as f:
    lines = np.loadtxt(f, delimiter=",")
dictLoad = np.asarray(lines).T

'''
