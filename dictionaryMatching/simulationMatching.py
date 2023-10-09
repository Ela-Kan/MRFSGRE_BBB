#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""------------------------------------------------------------------------
Matching of noisy simulations over multiple samples for statistical analysis 

Author: Emma Thomson
Year: 2020
Institution: Centre for Medical Image Computing: University College London
Email: e.thomson.19@ucl.ac.uk
------------------------------------------------------------------------"""

import numpy as np
import glob
import os 
from scipy.stats import iqr
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

scaler = MinMaxScaler()

#go up a folder
os.chdir("..")
print(os.getcwd())

"""-----------------------------FUNCTIONS----------------------------------"""
 
def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy) 
    #mi = normalized_mutual_info_score(list(x.T),list(y.T))
    nmi = mi /(entropy(x)*entropy(y))
    return nmi

#FOR NORMALISING DENSITY ARRAY
def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


"""----------------------------INPUTS---------------------------------"""

#number of noise levels used in generation
noi = 10
#number of samples at each noise level used in generation
samples = 50
#length of signal
acqlen = 500
#size of the dictionary - used for setting array size, can overestimate
no_entrys = 150
#number of dimensions in the dictioanry
parameters = 5

#Name of the dictioanry
dictionaryid = '2Dsim'
sampleid = dictionaryid

#load in dictionary 
#Load files
array = np.zeros((acqlen,no_entrys))
files = []

count = 0
sample = [0,0]
#rr = '*' +  str(repeat) + '.npy'
rr = '*_1.npy'
#Loading all signals

#for filename in glob.glob(os.path.join(str('/Users/emmathomson/Desktop/Local/Dictionaries/Dictionary' + dictionaryid + '/' + rr))):
for filename in glob.glob((str('./dictionaries/Dictionary' + dictionaryid + '/' + rr))):
#for filename in glob.glob(os.path.join(str('/Volumes/MyPassportforMac/Dictionaries_2022/Dictionaries/Dictionary' + dictionaryid + '/' + rr))):    
        with open(os.path.join(filename), 'r') as f:
            hold = np.load(filename)
            array[:,count] = np.squeeze(hold[:acqlen,0]/np.linalg.norm(hold[:acqlen,0]))
            files.append(filename)

        count += 1


res_ave = np.zeros([noi,parameters])
res_std = np.zeros([noi,parameters])
res_perc = np.zeros([noi,parameters])

#matching
for per in range(10,105,10): #range(5,105,5): #range(1,11,1): 
    for res in range(200,1700,100): #range(200,1700,100): #100000000000
        for T1t in [1300]: #range(600,1600,100): #range(1100,1550,50):
            for T1b in [1600]: #range(1500,2100,200):
                for multi in [100]: #range(80, 120, 10): #range(80, 125, 5):
                    res_est = np.zeros([parameters,samples])                                  
                    for noise in range(0,noi):
                            #for samp in range(0,samples):      
                            samp=49
                            correct_array = []
                            ind =[]
                            
                            #perc = sam[int(per)]
                            perc = per/10
                        
                            #Load the signal to match 
                            test_name = './Dictionaries/Dictionary' + sampleid + '/echo_' + str(T1t) + '_' + str(T1b) + '_' + str(res) + '_' + str(perc) + '_' + str(multi/100) + '_' + str(samp+1) + '.npy' 

                            hold =  np.load(test_name)
                            test_signal = hold[:acqlen,noise]
                            test_signal = test_signal/np.linalg.norm(test_signal)
                        
                            #test_signal = np.expand_dims(test_signal, axis=0)
                            #Calculate the dot product 
                            dot_sum = np.matmul(test_signal,array[:acqlen,:])


                            #Find max index 
                            dot_sum = dot_sum[dot_sum != 0]
                            max_index = np.argmax(dot_sum)
                            ind.append(max_index)
                            match_name = files[max_index]
                            strsplit = match_name.split('_')
                            try:
                                res_est[:,samp] = [float(strsplit[2]),float(strsplit[3]), float(strsplit[4]), float(strsplit[5]), float(strsplit[6])] #[:-4] 
                            except: 
                                res_est[:,samp] = [float(strsplit[1]),float(strsplit[2]), float(strsplit[3]), float(strsplit[4]), float(strsplit[5])] #[:-4] 
                   
                        #mode_calc, count = mode(res_est[:,:-1], axis=1)
                        #res_ave[noise,:] = np.squeeze(mode_calc)
                        res_ave[noise,:] = np.median(res_est[:,1:], axis=1)
                        res_std[noise,:] = iqr(res_est[:,1:], axis=1)
                    #res_perc[noise,:] = (np.mean(res_est, axis=1)-res)/res*100
                    
                    print('Input signal')
                    print(test_name)  
                    print ('Matched:')
                    print(str(res_ave[0,0]), str(res_ave[0,1]),
                          str(res_ave[0,2]), str(res_ave[0,3]), str(res_ave[0,4]))
                
                    pc = [res_ave, res_std]
                
                    file_name = './Dictionaries/SimulationMatching/Matching'+ sampleid + '/' + str(T1t) + '_' + str(T1b) + '_' + str(res) + '_' + str(perc) + '_' + str(multi/100) + '_pc.npy'
    
                    np.save(file_name, pc)
