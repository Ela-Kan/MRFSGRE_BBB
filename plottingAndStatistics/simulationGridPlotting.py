#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""------------------------------------------------------------------------

Plotting of simulation results/noise thresholds - in grid format 
    - Try colour grids and quiver plots 

Author: Emma Thomson
Year: 2023
Institution: Centre for Medical Image Computing: University College London
Email: e.thomson.19@ucl.ac.uk
------------------------------------------------------------------------"""

''' -----------------------------PACKAGES--------------------------------- '''

import numpy as np

import glob
import os
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times'
plt.rcParams['font.size'] = 55

import warnings
warnings.filterwarnings("ignore")

from palettable.colorbrewer.diverging import RdYlGn_6_r as colour
colourHold = colour.mpl_colormap 
colour_map = colourHold 

import time

t0 = time.time()

#go up a folder
os.chdir("..")

''' -----------------------------FUNCTIONS--------------------------------- '''


def sigmoid(x,L,k,x0): #m,c #L,k,x0
    func = 200*L/(1+np.exp(-k*(x-x0)))#+b
    #y = m*x+c
    return np.squeeze(func)

def find_nearest(yarray, xarray, value):
    array = np.asarray(yarray)
    idx=(np.abs(array-value)).argmin()
    return xarray[idx]

''' -----------------------------INPUTS--------------------------------- '''
#folder name
unique_id = '5Dsim'


#noise levels 
noise = 10
#dictionary repeats 
repeats = 50
dimensionality = 5 #2

#range of noises for grid plots
noise_range = [0.00, 0.01, 0.03, 0.06]

maptype = 'quiver' #'quantitative'

#number of isochromats dictionary is created with 
#need to sqrt
iso = np.sqrt(100)

patternMatchingPath = './Dictionaries/simulationMatching/Matching_'

''' -----------------------------SET LIMITS--------------------------------- '''

if dimensionality == 5: 
    #dimensionality of each dictionary dimension
    x1 = 6 #6 #T1t #6
    x2 = 3 #3 #T1b #3
    x3 = 15 #res
    x4 = 10 #perc
    x5 = 5 #5 #multi #5
else: 
    #dimensionality of each dictionary dimension
    x1 = 1 #T1t #6
    x2 = 1 #T1b #3
    x3 = 15 #res
    x4 = 10 #perc
    x5 = 1 #multi #5
    
#what percentages do you want to consider
percentages = np.array([range(0,10)])

#limit of successful matching 
limit_blood = 1
limit_res = 100

''' -----------------------------READ IN--------------------------------- '''

#empty arrays 
array_mean = np.zeros((x1,x2,x3,x4,x5,5,noise))
array_err = np.zeros((x1,x2,x3,x4,x5,5,noise))
files = []

if dimensionality == 5:
    mean_rep_array_blood = np.zeros([x1,x2,x3,x4,x5,noise]) #x1,x2 and x5 removed as they were singular dimensions
    std_rep_array_blood = np.zeros([x1,x2,x3,x4,x5,noise])
    mean_rep_array_res = np.zeros([x1,x2,x3,x4,x5,noise]) #x1,x2 and x5 removed as they were singular dimensions
    std_rep_array_res = np.zeros([x1,x2,x3,x4,x5,noise])

mean_rep_array_blood = np.zeros([x3,x4,noise]) #x1,x2 and x5 removed as they were singular dimensions
std_rep_array_blood = np.zeros([x3,x4,noise])
mean_rep_array_res = np.zeros([x3,x4,noise]) #x1,x2 and x5 removed as they were singular dimensions
std_rep_array_res = np.zeros([x3,x4,noise])
mean_rep_array_t1b = np.zeros([x3,x4,noise]) #x1,x2 and x5 removed as they were singular dimensions
std_rep_array_t1b = np.zeros([x3,x4,noise])
mean_rep_array_t1t = np.zeros([x3,x4,noise]) #x1,x2 and x5 removed as they were singular dimensions
std_rep_array_t1t = np.zeros([x3,x4,noise])
mean_rep_array_b1 = np.zeros([x3,x4,noise]) #x1,x2 and x5 removed as they were singular dimensions
std_rep_array_b1 = np.zeros([x3,x4,noise])


for noi in range(0,noise):
    #Load all file
    #for repeat in range(1,repeats+1,1):
        rr = '*.npy'
        for filename in glob.glob(os.path.join(str(patternMatchingPath + unique_id + '/'+ rr))):
                            with open(os.path.join(filename), 'r') as f: 
                                split = filename.split('/')
                                split = split[4].split('_')
                                if dimensionality == 5: 
                                    x1fill = (int(split[0]) - 1000)/200 #T1t
                                    x2fill = (int(split[1]) - 1500)/200 #T1b
                                    x3fill = (int(split[2]) - 200)/100 #res int(float(split[2]))/100000000000 
                                    x4fill = int(float(split[3]))-1
                                    x5fill = (round(float(split[4])*100,0)-80)/10#multi
                                else: 
                                    x1fill = (int(split[0]) - 1300)/200 #T1t
                                    x2fill = (int(split[1]) - 1600)/200 #T1b
                                    x3fill = (int(split[2]) - 200)/100 #res int(float(split[2]))/100000000000
                                    x4fill = int(float(split[3]))-1
                                    x5fill = (round(float(split[4])*100,0)-100)/10#multi
                                
                                array_mean[int(x1fill),int(x2fill),int(x3fill),\
                                      int(x4fill),int(x5fill),:,:] = \
                                    np.expand_dims(np.squeeze(np.load(filename))[0,noi,:].T, axis=1)#,int(x5fill)
                                array_err[int(x1fill),int(x2fill),int(x3fill),\
                                      int(x4fill),int(x5fill),:,:] = \
                                    np.expand_dims(np.squeeze(np.load(filename))[1,noi,:].T, axis=1)#,int(x5fill)
                                files.append([int(x1fill),int(x2fill),int(x3fill),\
                                     int(x4fill), int(x5fill), filename]) #int(x5fill)
                                
        perc_mean = np.squeeze(array_mean[:,:,:,:,:,:,noi])  
        perc_err = np.squeeze(array_err[:,:,:,:,:,:,noi]) 

        if dimensionality == 5:
            mean_rep_blood = np.mean(np.mean(np.mean(perc_mean[:,:,:,:,:,3],axis=0),axis=0),axis=-1)
            std_rep_blood = np.mean(np.mean(np.mean(perc_err[:,:,:,:,:,3],axis=0),axis=0),axis=-1) 
            mean_rep_array_blood[:,:,noi]= mean_rep_blood 
            std_rep_array_blood[:,:,noi] = std_rep_blood 
    
            mean_rep_res = np.mean(np.mean(np.mean(perc_mean[:,:,:,:,:,2],axis=0),axis=0),axis=-1)
            std_rep_res = np.mean(np.mean(np.mean(perc_err[:,:,:,:,:,2],axis=0),axis=0),axis=-1)
            mean_rep_array_res[:,:,noi]= mean_rep_res
            std_rep_array_res[:,:,noi] = std_rep_res 
            
            mean_rep_t1b = np.mean(np.mean(np.mean(perc_mean[:,:,:,:,:,1],axis=0),axis=0),axis=-1) 
            std_rep_t1b = np.mean(np.mean(np.mean(perc_err[:,:,:,:,:,1],axis=0),axis=0),axis=-1)
            mean_rep_array_t1b[:,:,noi]= mean_rep_t1b 
            std_rep_array_t1b[:,:,noi] = std_rep_t1b 

            mean_rep_t1t = np.mean(np.mean(np.mean(perc_mean[:,:,:,:,:,0],axis=0),axis=0),axis=-1)
            std_rep_t1t = np.mean(np.mean(np.mean(perc_err[:,:,:,:,:,0],axis=0),axis=0),axis=-1)
            mean_rep_array_t1t[:,:,noi]= mean_rep_t1t
            std_rep_array_t1t[:,:,noi] = std_rep_t1t 

            mean_rep_b1 = np.mean(np.mean(np.mean(perc_mean[:,:,:,:,:,4],axis=0),axis=0),axis=-1)
            std_rep_b1 = np.mean(np.mean(np.mean(perc_err[:,:,:,:,:,4],axis=0),axis=0),axis=-1)
            mean_rep_array_b1[:,:,noi]= mean_rep_b1
            std_rep_array_b1[:,:,noi] = std_rep_b1 
            
        else:   
 
            mean_rep_blood = perc_mean[:,:,3]
            
            std_rep_blood = perc_err[:,:,3] 
            mean_rep_array_blood[:,:,noi]= mean_rep_blood 
            std_rep_array_blood[:,:,noi] = std_rep_blood 

            mean_rep_res = perc_mean[:,:,2] 
            std_rep_res = perc_err[:,:,2]

            mean_rep_array_res[:,:,noi]= mean_rep_res
            std_rep_array_res[:,:,noi] = std_rep_res

'''-----------------------------QUIVER PLOTS----------------------------------'''

# Creating arrow
xx = np.arange(1, 11, 1)
yy = np.arange(200, 1700, 100)
X, Y = np.meshgrid(xx, yy)

#Arrow spcifics
wid = 0.015
sca = 2

fig, ax = plt.subplots(1,4, constrained_layout=True)
fig.set_size_inches(85,26, forward=True)
plt.rcParams.update({'font.size': 90})
plt.rcParams['xtick.labelsize'] =90
plt.rcParams['ytick.labelsize'] = 90


fig.supylabel('Ground Truth Residence Time (ms)')

fig.supxlabel('Ground Truth Blood Volume (%)')


for noise_plot in noise_range: 
    
    ind = noise_range.index(noise_plot) 
    index = int(noise_plot*100)
    
    u = mean_rep_array_blood[:,:,index] - X
    v = mean_rep_array_res[:,:,index] - Y
    
    u2 = std_rep_array_blood[:,:,index] 
    v2 = std_rep_array_res[:,:,index] 
    
    color = (v2/1500 + u2/10)/2*100

    Q = ax[ind].quiver(X, Y, u, v, color, width=wid, angles='xy',
                       scale_units='xy', scale=1, cmap=colour_map)
    Q.set_clim(0,60)
    
    try:
        ax[ind].set_title(' SNR = ' + str(np.round(13.04/(100*noise_plot),1)) + " \n"  + r'$\sigma_G =$' + str(noise_plot) + '%')
        #ax[ind].set_title(r'$\sigma_G =$' + str(noise_plot) + '%')
    except: 
        ax[ind].set_title(r' SNR = $\infty$' + " \n" + r'$\sigma_G =$' + str(noise_plot) + '%')
    ax[ind].set_ylim([200,1600])

cbar = plt.colorbar(Q, label='IQR of match over \n 50 repeats [% of dictionary range]')

'''-----------------------------QUIVER PLOTS----------------------------------'''
x_tile = np.tile(np.expand_dims(X,2),[1,1,noise])
y_tile = np.tile(np.expand_dims(Y,2),[1,1,noise])

meannub =  np.mean(abs(mean_rep_array_blood-x_tile).reshape(150,10),0)
stdnub = np.std(abs(mean_rep_array_blood-x_tile).reshape(150,10),0)

meantaub =  np.mean(abs(mean_rep_array_res-y_tile).reshape(150,10),0)
stdtaub =  np.std(abs(mean_rep_array_res-y_tile).reshape(150,10),0)

data = np.array([meannub, stdnub, meantaub, stdtaub])

meannub =  np.mean(abs(std_rep_array_blood).reshape(150,10),0)
stdnub = np.std(abs(std_rep_array_blood).reshape(150,10),0)

meantaub =  np.mean(abs(std_rep_array_res).reshape(150,10),0)
stdtaub = np.std(abs(std_rep_array_res).reshape(150,10),0)

dataIQR = np.array([meannub, stdnub, meantaub, stdtaub])


'''-----------------------------THRESHOLD PLOTS----------------------------------'''

noisePlotting = (np.array(range(1,noise,1)))/100 #13.04/(np.array(range(1,noise,1)))


#ax = plt.subplot2grid((80,32), (0,0))
fig, ax = plt.subplots(1,2, constrained_layout=True)
fig.set_size_inches(15,5, forward=True)
plt.rcParams.update({'font.size': 20})
plt.rcParams['xtick.labelsize'] =20
plt.rcParams['ytick.labelsize'] = 20
#gs = gridspec.GridSpec(, width_ratios=[[1, 1, 2], [1, 1, 2], [1, 1, 2], [1, 1, 2]]) 
#plt.subplots_adjust(top=0.9)
#fig.supylabel('Ground Truth Residence Time (ms)')
ax[0].set_xlabel(r'Noise ($\sigma_G$)')#, ylabel='Ground Truth Residence Time (ms)')
ax[1].set_xlabel(r'Noise ($\sigma_G$)')
#ig.supxlabel('log(SNR)')
ax[0].errorbar(noisePlotting, dataIQR[0,1:], yerr=dataIQR[1,1:],  capsize=7, fmt="o-", c="black")
ax[0].set_ylabel('Mean Error in Median \n Matched Value for ' r'$\nu_b$ [%]')
ax[0].plot([0,0.1], [0.5,0.5], 'r--')
#ax[0].plot([0.07,0.07], [-0.3,0.5], 'r--')
ax[0].set_ylim(-0.3,2.5)
ax[0].set_xlim(0,0.1)
ax[1].errorbar(noisePlotting, dataIQR[2,1:], yerr=dataIQR[3,1:],  capsize=7, fmt="o-", c="black")
ax[1].set_ylabel('Mean Error in Median \n Matched Value for ' r'$\tau_b$ [ms]')
ax[1].plot([0,0.1], [50,50], 'r--')
#ax[1].plot([0.04,0.04], [-100,50], 'r--')
ax[1].set_ylim(-100,700)
ax[1].set_xlim(0,0.1)
