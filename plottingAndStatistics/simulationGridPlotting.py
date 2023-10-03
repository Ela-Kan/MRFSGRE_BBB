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
import re 
import glob
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.optimize import curve_fit
from scipy.stats import iqr


plt.rcParams['font.family'] = 'Times'
plt.rcParams['font.size'] = 55

'''
plt.rcParams['xtick.labelsize'] =40
plt.rcParams['ytick.labelsize'] = 40
'''
#import shutup
#shutup.please()
import warnings
warnings.filterwarnings("ignore")

#import fsleyes
#fsleyes.colourmaps.scanColourMaps()
#os.chdir('/opt/anaconda3/lib/python3.8/site-packages/fsleyes/assets/colourmaps/brain_colours/')
#fsleyes.colourmaps.init()
import cmcrameri.cm as cmc
#from cmcrameri import show_cmapspip
#show_cmaps()
#from palettable.colorbrewer.sequential import OrRd_5 as colour
#from palettable.cmocean.sequential import Amp_5 as colour
from palettable.colorbrewer.diverging import RdYlGn_6_r as colour
colourHold = colour.mpl_colormap #cmc.imola_r #colour.mpl_colormap
colour_map = colourHold #fsleyes.colourmaps.getColourMap('brain_colours_redgray_iso')

import time

t0 = time.time()

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
unique_id = 'ISMRMlargeHold'


#noise levels 
noise = 10
#dictionary repeats 
repeats = 50
dimensionality = 5

#range of noises for grid plots
noise_range = [0.00, 0.01, 0.03, 0.09]

#save plots or nah
save = 0

maptype = 'quiver' #'quantitative' #'no


#number of isochromats dictionary is created with 
#need to sqrt
iso = np.sqrt(100)
#sampling density
sampling = 'uniform'#'uniform' #'gaussian'

patternMatchingPath = '/Users/emmathomson/Desktop/Local/Pattern_Matching_Paper/Pattern_Matching_'

''' -----------------------------SET LIMITS--------------------------------- '''

if dimensionality == 5: 
    #dimensionality of each dictionary dim ension
    x1 = 6 #6 #T1t #6
    x2 = 3 #3 #T1b #3
    x3 = 15 #res
    x4 = 10 #perc
    x5 = 5 #5 #multi #5
else: 
    #dimensionality of each dictionary dim ension
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

#array = np.zeros((10,14,21))

#empty arrays 
array_mean = np.zeros((x1,x2,x3,x4,x5,5,noise))
array_err = np.zeros((x1,x2,x3,x4,x5,5,noise))
files = []
if sampling == 'gaussian':
    samples = np.load('/Users/emmathomson/Desktop/samples.npy')
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
                                split = split[7].split('_')
                                if dimensionality == 5: 
                                    x1fill = (int(split[0]) - 1000)/200 #T1t
                                    x2fill = (int(split[1]) - 1500)/200 #T1b
                                    x3fill = (int(split[2]) - 200)/100 #res int(float(split[2]))/100000000000 
                                    x5fill = (round(float(split[4])*100,0)-80)/10#multi
                                else: 
                                    x1fill = (int(split[0]) - 1300)/200 #T1t
                                    x2fill = (int(split[1]) - 1600)/200 #T1b
                                    x3fill = (int(split[2]) - 200)/100 #res int(float(split[2]))/100000000000
                                    x5fill = (round(float(split[4])*100,0)-100)/10#multi
                                if sampling  == 'gaussian':
                                    x4fill = np.where(samples == np.float64(split[3])) #perc 
                                    x4fill = int(x4fill[0])
                                else: 
                                    x4fill = int(float(split[3]))-1
                                
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
            #mean_rep_blood = perc_mean[:,:,3] #np.median(np.squeeze(perc[:,:,3,:]),axis=2)
            #std_rep_blood = perc_err[:,:,3] #iqr(np.squeeze(perc[:,:,3,:]),axis=2)
            #mean_rep_blood = np.mean(np.mean(np.mean(perc_mean[:,:,:,:,:,3],axis=0),axis=0),axis=-1) #np.median(np.squeeze(perc[:,:,3,:]),axis=2)
            #std_rep_blood = np.mean(np.mean(np.mean(perc_err[:,:,:,:,:,3],axis=0),axis=0),axis=-1) #iqr(np.squeeze(perc[:,:,3,:]),axis=2)
            mean_rep_blood = np.mean(np.mean(np.mean(perc_mean[:,:,:,:,:,3],axis=0),axis=0),axis=-1) #np.median(np.squeeze(perc[:,:,3,:]),axis=2)
            std_rep_blood = np.mean(np.mean(np.mean(perc_err[:,:,:,:,:,3],axis=0),axis=0),axis=-1) #iqr(np.squeeze(perc[:,:,3,:]),axis=2)
            mean_rep_array_blood[:,:,noi]= mean_rep_blood #np.expand_dims(mean_rep,axis=2)
            std_rep_array_blood[:,:,noi] = std_rep_blood #np.expand_dims(std_rep,axis=2)
    
      
            #mean_rep_res = perc_mean[:,:,2] #np.median(np.squeeze(perc[:,:,2,:]),axis=2)
            #std_rep_res = perc_err[:,:,2] #iqr(np.squeeze(perc[:,:,2,:]),axis=2) 
            mean_rep_res = np.mean(np.mean(np.mean(perc_mean[:,:,:,:,:,2],axis=0),axis=0),axis=-1)#np.median(np.squeeze(perc[:,:,2,:]),axis=2)
            std_rep_res = np.mean(np.mean(np.mean(perc_err[:,:,:,:,:,2],axis=0),axis=0),axis=-1)#iqr(np.squeeze(perc[:,:,2,:]),axis=2) 
            mean_rep_array_res[:,:,noi]= mean_rep_res#np.expand_dims(mean_rep,axis=2)
            std_rep_array_res[:,:,noi] = std_rep_res #np.expand_dims(std_rep,axis=2)
            
            
            #mean_rep_blood = perc_mean[:,:,3] #np.median(np.squeeze(perc[:,:,3,:]),axis=2)
            #std_rep_blood = perc_err[:,:,3] #iqr(np.squeeze(perc[:,:,3,:]),axis=2)
            mean_rep_t1b = np.mean(np.mean(np.mean(perc_mean[:,:,:,:,:,1],axis=0),axis=0),axis=-1) #np.median(np.squeeze(perc[:,:,3,:]),axis=2)
            std_rep_t1b = np.mean(np.mean(np.mean(perc_err[:,:,:,:,:,1],axis=0),axis=0),axis=-1) #iqr(np.squeeze(perc[:,:,3,:]),axis=2)
            mean_rep_array_t1b[:,:,noi]= mean_rep_t1b #np.expand_dims(mean_rep,axis=2)
            std_rep_array_t1b[:,:,noi] = std_rep_t1b #np.expand_dims(std_rep,axis=2)
    
      
            #mean_rep_res = perc_mean[:,:,2] #np.median(np.squeeze(perc[:,:,2,:]),axis=2)
            #std_rep_res = perc_err[:,:,2] #iqr(np.squeeze(perc[:,:,2,:]),axis=2) 
            mean_rep_t1t = np.mean(np.mean(np.mean(perc_mean[:,:,:,:,:,0],axis=0),axis=0),axis=-1)#np.median(np.squeeze(perc[:,:,2,:]),axis=2)
            std_rep_t1t = np.mean(np.mean(np.mean(perc_err[:,:,:,:,:,0],axis=0),axis=0),axis=-1)#iqr(np.squeeze(perc[:,:,2,:]),axis=2) 
            mean_rep_array_t1t[:,:,noi]= mean_rep_t1t#np.expand_dims(mean_rep,axis=2)
            std_rep_array_t1t[:,:,noi] = std_rep_t1t #np.expand_dims(std_rep,axis=2)
            
            #mean_rep_res = perc_mean[:,:,2] #np.median(np.squeeze(perc[:,:,2,:]),axis=2)
            #std_rep_res = perc_err[:,:,2] #iqr(np.squeeze(perc[:,:,2,:]),axis=2) 
            mean_rep_b1 = np.mean(np.mean(np.mean(perc_mean[:,:,:,:,:,4],axis=0),axis=0),axis=-1)#np.median(np.squeeze(perc[:,:,2,:]),axis=2)
            std_rep_b1 = np.mean(np.mean(np.mean(perc_err[:,:,:,:,:,4],axis=0),axis=0),axis=-1)#iqr(np.squeeze(perc[:,:,2,:]),axis=2) 
            mean_rep_array_b1[:,:,noi]= mean_rep_b1#np.expand_dims(mean_rep,axis=2)
            std_rep_array_b1[:,:,noi] = std_rep_b1 #np.expand_dims(std_rep,axis=2)
            
        else:   
            #mean_rep_blood = perc_mean[:,:,3] #np.median(np.squeeze(perc[:,:,3,:]),axis=2)
            #std_rep_blood = perc_err[:,:,3] #iqr(np.squeeze(perc[:,:,3,:]),axis=2)
            mean_rep_blood = perc_mean[:,:,3] #np.median(np.squeeze(perc[:,:,3,:]),axis=2)
            std_rep_blood = perc_err[:,:,3] #iqr(np.squeeze(perc[:,:,3,:]),axis=2)
            #mean_rep_blood = np.median(np.squeeze(perc_mean[:,:,3]),axis=2)
            #std_rep_blood = iqr(np.squeeze(perc_err[:,:,3]),axis=2)
            mean_rep_array_blood[:,:,noi]= mean_rep_blood #np.expand_dims(mean_rep,axis=2)
            std_rep_array_blood[:,:,noi] = std_rep_blood #np.expand_dims(std_rep,axis=2)
    
      
            #mean_rep_res = perc_mean[:,:,2] #np.median(np.squeeze(perc[:,:,2,:]),axis=2)
            #std_rep_res = perc_err[:,:,2] #iqr(np.squeeze(perc[:,:,2,:]),axis=2) 
            mean_rep_res = perc_mean[:,:,2] #np.median(np.squeeze(perc[:,:,2,:]),axis=2)
            std_rep_res = perc_err[:,:,2] #iqr(np.squeeze(perc[:,:,2,:]),axis=2) 
            #mean_rep_res = np.median(np.squeeze(perc_mean[:,:,2]),axis=2)
            #std_rep_res = iqr(np.squeeze(perc_err[:,:,2]),axis=2) 
            mean_rep_array_res[:,:,noi]= mean_rep_res#np.expand_dims(mean_rep,axis=2)
            std_rep_array_res[:,:,noi] = std_rep_res #np.expand_dims(std_rep,axis=2)


''' -----------------------------PLOTTING-------------------------------- '''

unique_id = unique_id + '_' + str(repeats)

# generate 2 2d grids for the x & y bounds
#y, x = np.mgrid[slice(200, 1700, 100),slice(1, 11, 1)]
y, x = np.mgrid[slice(200, 1700, 100),slice(1, x4+1, 1)]
if sampling == 'gaussian':
    x = np.tile(np.float64(samples).T,(x3,1))
    
ground_truth = x

#set limits for the different grid types
z_min_acc_blood, z_max_acc_blood = -9, 9
z_min_prec_blood, z_max_prec_blood = 0,10

z_min_acc_res, z_max_acc_res = -1500,1500
z_min_prec_res, z_max_prec_res = 0,1600

'''
    PLOT AS SINGLE FIGURE
'''

###
###     PLOTTING BLOOD ACCURACY
###

#ax = plt.subplot2grid((80,32), (0,0))
fig, ax = plt.subplots(4,4, constrained_layout=True)
fig.set_size_inches(33,30, forward=True)
plt.rcParams.update({'font.size': 50})
#gs = gridspec.GridSpec(, width_ratios=[[1, 1, 2], [1, 1, 2], [1, 1, 2], [1, 1, 2]]) 

#plt.subplots_adjust(top=0.9)

fig.supylabel('Ground Truth Residence Time (ms)')

fig.supxlabel('Ground Truth Blood Volume (%)')#, ylabel='Ground Truth Residence Time (ms)')


'''

#fig.suptitle('title')
ax0 = plt.subplot2grid((80, 32), (0, 0), colspan=10, rowspan=20)
ax1 = plt.subplot2grid((80, 32), (0, 10), colspan=10, rowspan =20)
ax2 = plt.subplot2grid((80, 32), (0, 20), colspan=12, rowspan=20)

## ax0

ind = noise_range.index(noise_range[0]) 
index = int(noise_range[0]*100)

ground_truth = x
data_acc_blood_hold = mean_rep_array_blood[:,:,index]
data_acc_blood = data_acc_blood_hold - ground_truth

c = ax0.pcolormesh(x, y, data_acc_blood, cmap='RdBu_r', vmin=z_min_acc_blood, vmax=z_max_acc_blood, shading='auto')
ax0.set(xlabel='Ground Truth Blood Volume (%)', ylabel='Ground Truth Residence Time (ms)')
#plt.rcParams.update({'font.size': 30})
ax0.set_title('n =' + str(noise_range[0]))
#plt.rcParams.update({'font.size': 26})

## ax1

ind = noise_range.index(noise_range[1]) 
index = int(noise_range[1]*100)

data_acc_blood_hold = mean_rep_array_blood[:,:,index]
data_acc_blood = data_acc_blood_hold - ground_truth

c = ax1.pcolormesh(x, y, data_acc_blood, cmap='RdBu_r', vmin=z_min_acc_blood, vmax=z_max_acc_blood, shading='auto')
ax1.set(xlabel='Ground Truth Blood Volume (%)', ylabel='Ground Truth Residence Time (ms)')
#plt.rcParams.update({'font.size': 30})
ax1.set_title('n =' + str(noise_range[1]))
#plt.rcParams.update({'font.size': 26})

## ax2

ind = noise_range.index(noise_range[2]) 
index = int(noise_range[2]*100)

data_acc_blood_hold = mean_rep_array_blood[:,:,index]
data_acc_blood = data_acc_blood_hold - ground_truth

c = ax2.pcolormesh(x, y, data_acc_blood, cmap='RdBu_r', vmin=z_min_acc_blood, vmax=z_max_acc_blood, shading='auto')
ax2.set(xlabel='Ground Truth Blood Volume (%)', ylabel='Ground Truth Residence Time (ms)')
#plt.rcParams.update({'font.size': 30})
ax2.set_title('n =' + str(noise_range[2]))
#plt.rcParams.update({'font.size': 26})
#ax.subplots_adjust(right = 1.15)
ax2.colorbar(c, ax=ax2, label='Deviation of Median Matched \n Blood Volume from Ground Truth (%)')
'''


for noise_plot in noise_range: 
    
    ind = noise_range.index(noise_plot) 
    index = int(noise_plot*100)

    #data for plotting - blood matching accuracy
    ground_truth = x
    data_acc_blood_hold = mean_rep_array_blood[:,:,index]
    data_acc_blood = data_acc_blood_hold - ground_truth
    print(np.mean(abs(data_acc_blood)), ' + ' , np.std(abs(data_acc_blood)))
    
    c = ax[0,ind].pcolormesh(x, y, data_acc_blood, cmap='bwr', vmin=z_min_acc_blood, vmax=z_max_acc_blood, shading='auto')
    #ax[0,ind].set(xlabel='Ground Truth Blood Volume (%)', ylabel='Ground Truth Residence Time (ms)')
    #plt.rcParams.update({'font.size': 34})
    ax[0,ind].set_title(r'$ \sigma_G =$' + str(noise_plot))
    #plt.rcParams.update({'font.size': 28})


#fig = plt.gcf()
#plt.rcParams.update({'font.size': 48})
#plt.rcParams['figure.constrained_layout.use'] = True
#fig.subplots_adjust(right = 1.5)
#plt.rcParams['figure.constrained_layout.use'] = True

fig.colorbar(c, ax=ax[0,:], location='right', anchor=(4, 0.3), label='Deviation of Median \n Matched Blood  \n Volume from Ground \n Truth (%)')


###
###     PLOTTING BLOOD PRECISION
###

for noise_plot in noise_range: 
    
    ind = noise_range.index(noise_plot) 
    index = int(noise_plot*100)

    #data for plotting - blood matching precision
    data_prec_blood = std_rep_array_blood[:,:,index]  
    #afmhot_r
    c = ax[1,ind].pcolor(x, y, data_prec_blood, cmap='afmhot_r', vmin=z_min_prec_blood, vmax=z_max_prec_blood, shading = 'auto')

    #ax[1,ind].set(xlabel='Ground Truth Blood Volume (%)', ylabel='Ground Truth Residence Time (ms)')
    #ax[1,ind].set_title('n =' + str(noise_plot))
 

#plt.rcParams.update({'font.size': 48})
#fig.subplots_adjust(right = 1.5)
fig.colorbar(c, ax=ax[1,:], location='right', anchor=(4, 0.3), label='IQR of Matched \n Blood Volume (%)') 
#fig.subplots_adjust(right = 1)

#fig.tight_layout()

###
###     PLOTTING RESDIENCE ACCURACY
###
for noise_plot in noise_range: 
    
    ind = noise_range.index(noise_plot) 
    index = int(noise_plot*100)

    #data for plotting - residence time matching accuracy
    ground_truth = y
    data_acc_res_hold = mean_rep_array_res[:,:,index]
    std_acc_res_hold = std_rep_array_res[:,:,index]
    data_acc_res = data_acc_res_hold - ground_truth
    print(np.mean(abs(data_acc_res)), ' + ' , np.mean(abs(std_acc_res_hold)))
 
    if maptype == 'quantitative':
        c = ax[2,ind].pcolor(x, y, data_acc_res, cmap='Pastel1_r', vmin=z_min_acc_res, vmax=z_max_acc_res, shading = 'auto')
    else:
        c = ax[2,ind].pcolor(x, y, data_acc_res, cmap='bwr', vmin=z_min_acc_res, vmax=z_max_acc_res, shading = 'auto')
    
    #ax[2,ind].set(xlabel='Ground Truth Blood Volume (%)', ylabel='Ground Truth Residence Time (ms)')
    #ax[2,ind].set_title('n =' + str(noise_plot))

 
#fig = plt.gcf()
#plt.rcParams.update({'font.size': 48})
#fig.subplots_adjust(right = 1.15)
fig.colorbar(c, ax=ax[2,:], location='right', anchor=(4, 0.3), label='Deviation of Median \n Matched Residence  \n Time from Ground  \n Truth (ms)')
#fig.subplots_adjust(right = 1.15)


#plt.tight_layout(pad=0.4, w_pad=1, h_pad=0.5)


###
###     PLOTTING BLOOD PRECISION
###

for noise_plot in noise_range: 
    
    ind = noise_range.index(noise_plot) 
    index = int(noise_plot*100)

    #data for plotting - residence time matching precision
    data_prec_res = std_rep_array_res[:,:,index]  
 
    if maptype == 'quantitative':
        c = ax[3,ind].pcolor(x, y, data_prec_res, cmap='Pastel1_r', vmin=z_min_prec_res, vmax=z_max_prec_res, shading='auto')
    else:
        c = ax[3,ind].pcolor(x, y, data_prec_res, cmap='afmhot_r', vmin=z_min_prec_res, vmax=z_max_prec_res, shading='auto')
    #ax[3,ind].set(xlabel='Ground Truth Blood Volume (%)', ylabel='Ground Truth Residence Time (ms)')
    #ax[3,ind].set_title('n =' + str(noise_plot))
 
#plt.rcParams.update({'font.size': 48})3
#fig.subplots_adjust(right = 1.15)
fig.colorbar(c, ax=ax[3,:], location='right', anchor=(4, 0.3), label='IQR of Matched \n Residence Time  \n (ms)') 
        
#plt.tight_layout(pad=1, w_pad=2, h_pad=1)


'''

fig,ax = plt.subplots(1,1)
signal1 = np.load('/Users/emmathomson/Desktop/Dictionaries/DictionarybSSFP/echo_1300_1600_1600_10.0_1.0_50.npy')
#signal1 = signal1/np.max(signal1[:,0,0])

plt.plot(signal1[:,1,0], linewidth=2, label = r'$\sigma_G = 0.03$')
plt.plot(signal1[:,0,0], label = r'$\sigma_G = 0.00$')

ax.set_xlabel('TR index')
ax.set_ylabel('Normalised Signal')
fig.set_size_inches(11,8, forward=True)
plt.legend()
plt.rcParams.update({'font.size': 24})

'''

if dimensionality == 5: 
    

    #set limits for the different grid types
    z_min_acc_t1t, z_max_acc_t1t = -2000, 2000
    z_min_prec_t1t, z_max_prec_t1t = 0,2000
    
    z_min_acc_t1b, z_max_acc_t1b = -1000,1000
    z_min_prec_t1b, z_max_prec_t1b = 0,1000

    '''
        PLOT AS SINGLE FIGURE
    '''
    
    ###
    ###     PLOTTING BLOOD ACCURACY
    ###
    
    #ax = plt.subplot2grid((80,32), (0,0))
    fig, ax = plt.subplots(4,4, constrained_layout=True)
    fig.set_size_inches(33,30, forward=True)
    plt.rcParams.update({'font.size': 50})
    #gs = gridspec.GridSpec(, width_ratios=[[1, 1, 2], [1, 1, 2], [1, 1, 2], [1, 1, 2]]) 
    
    #plt.subplots_adjust(top=0.9)
    
    fig.supylabel('Ground Truth Residence Time (ms)')
    
    fig.supxlabel('Ground Truth Blood Volume (%)')#, ylabel='Ground Truth Residence Time (ms)')


    for noise_plot in noise_range: 
    
        ind = noise_range.index(noise_plot) 
        index = int(noise_plot*100)
    
        #data for plotting - blood matching accuracy
        ground_truth = 1500
        data_acc_blood_hold = mean_rep_array_t1t[:,:,index]
        data_acc_blood = data_acc_blood_hold - ground_truth
        print(np.mean(abs(data_acc_blood)))
        
        c = ax[0,ind].pcolormesh(x, y, data_acc_blood, cmap='bwr', vmin=z_min_acc_t1t, vmax=z_max_acc_t1t, shading='auto')
        #ax[0,ind].set(xlabel='Ground Truth Blood Volume (%)', ylabel='Ground Truth Residence Time (ms)')
        #plt.rcParams.update({'font.size': 34})
        ax[0,ind].set_title(r'$ \sigma_G =$' + str(noise_plot))
        #plt.rcParams.update({'font.size': 28})


    #fig = plt.gcf()
    #plt.rcParams.update({'font.size': 48})
    #plt.rcParams['figure.constrained_layout.use'] = True
    #fig.subplots_adjust(right = 1.5)
    #plt.rcParams['figure.constrained_layout.use'] = True
    
    fig.colorbar(c, ax=ax[0,:], location='right', anchor=(4, 0.3), label='Deviation of Median \n Matched Tissue \n T1 from Ground \n Truth (ms)')
    
    
    ###
    ###     PLOTTING BLOOD PRECISION
    ###
    
    for noise_plot in noise_range: 
        
        ind = noise_range.index(noise_plot) 
        index = int(noise_plot*100)
    
        #data for plotting - blood matching precision
        data_prec_blood = std_rep_array_t1t[:,:,index]  
        #afmhot_r
        c = ax[1,ind].pcolor(x, y, data_prec_blood, cmap='afmhot_r', vmin=z_min_prec_t1t, vmax=z_max_prec_t1t, shading = 'auto')
        
        #ax[1,ind].set(xlabel='Ground Truth Blood Volume (%)', ylabel='Ground Truth Residence Time (ms)')
        #ax[1,ind].set_title('n =' + str(noise_plot))
     
    
    #plt.rcParams.update({'font.size': 48})
    #fig.subplots_adjust(right = 1.5)
    fig.colorbar(c, ax=ax[1,:], location='right', anchor=(4, 0.3), label='IQR of Matched \n Tissue T1 (ms)') 
    #fig.subplots_adjust(right = 1)
    
    #fig.tight_layout()
    
    ###
    ###     PLOTTING RESDIENCE ACCURACY
    ###
    for noise_plot in noise_range: 
        
        ind = noise_range.index(noise_plot) 
        index = int(noise_plot*100)
    
        #data for plotting - residence time matching accuracy
        ground_truth = 1700
        data_acc_res_hold = mean_rep_array_t1b[:,:,index]
        data_acc_res = data_acc_res_hold - ground_truth
        print(np.mean(abs(data_acc_res)))
     
        c = ax[2,ind].pcolor(x, y, data_acc_res, cmap='bwr', vmin=z_min_acc_t1b, vmax=z_max_acc_t1b, shading = 'auto')
        
        #ax[2,ind].set(xlabel='Ground Truth Blood Volume (%)', ylabel='Ground Truth Residence Time (ms)')
        #ax[2,ind].set_title('n =' + str(noise_plot))
    
     
    #fig = plt.gcf()
    #plt.rcParams.update({'font.size': 48})
    #fig.subplots_adjust(right = 1.15)
    fig.colorbar(c, ax=ax[2,:], location='right', anchor=(4, 0.3), label='Deviation of Median \n Matched Blood  \n T1 from Ground  \n Truth (ms)')
    #fig.subplots_adjust(right = 1.15)
    
    
    #plt.tight_layout(pad=0.4, w_pad=1, h_pad=0.5)
    
    
    ###
    ###     PLOTTING BLOOD PRECISION
    ###
    
    for noise_plot in noise_range: 
        
        ind = noise_range.index(noise_plot) 
        index = int(noise_plot*100)
    
        #data for plotting - residence time matching precision
        data_prec_res = std_rep_array_t1b[:,:,index]  
     
    
        c = ax[3,ind].pcolor(x, y, data_prec_res, cmap='afmhot_r', vmin=z_min_prec_t1b, vmax=z_max_prec_t1b, shading='auto')
        #ax[3,ind].set(xlabel='Ground Truth Blood Volume (%)', ylabel='Ground Truth Residence Time (ms)')
        #ax[3,ind].set_title('n =' + str(noise_plot))
     
    #plt.rcParams.update({'font.size': 48})3
    #fig.subplots_adjust(right = 1.15)
    fig.colorbar(c, ax=ax[3,:], location='right', anchor=(4, 0.3), label='IQR of Matched \n Blood T1 (ms)') 
            
    #plt.tight_layout(pad=1, w_pad=2, h_pad=1)
    
    #set limits for the different grid types
    z_min_acc_b1, z_max_acc_b1 = -1.2, 1.2
    z_min_prec_b1, z_max_prec_b1 = 0,1.2

        
    ###
    ###     PLOTTING BLOOD ACCURACY
    ###
    
    #ax = plt.subplot2grid((80,32), (0,0))
    fig, ax = plt.subplots(2,4, constrained_layout=True)
    fig.set_size_inches(33,15, forward=True)
    plt.rcParams.update({'font.size': 50})
    #gs = gridspec.GridSpec(, width_ratios=[[1, 1, 2], [1, 1, 2], [1, 1, 2], [1, 1, 2]]) 
    
    #plt.subplots_adjust(top=0.9)
    
    fig.supylabel('Ground Truth Residence Time (ms)')
    
    fig.supxlabel('Ground Truth Blood Volume (%)')#, ylabel='Ground Truth Residence Time (ms)')


    for noise_plot in noise_range: 
    
        ind = noise_range.index(noise_plot) 
        index = int(noise_plot*100)
    
        #data for plotting - blood matching accuracy
        ground_truth = 1
        data_acc_blood_hold = mean_rep_array_b1[:,:,index]
        data_acc_blood = data_acc_blood_hold - ground_truth
        
        c = ax[0,ind].pcolormesh(x, y, data_acc_blood, cmap='bwr', vmin=z_min_acc_b1, vmax=z_max_acc_b1, shading='auto')
        #ax[0,ind].set(xlabel='Ground Truth Blood Volume (%)', ylabel='Ground Truth Residence Time (ms)')
        #plt.rcParams.update({'font.size': 34})
        ax[0,ind].set_title(r'$ \sigma_G =$' + str(noise_plot))
        #plt.rcParams.update({'font.size': 28})
    
    
    #fig = plt.gcf()
    #plt.rcParams.update({'font.size': 48})
    #plt.rcParams['figure.constrained_layout.use'] = True
    #fig.subplots_adjust(right = 1.5)
    #plt.rcParams['figure.constrained_layout.use'] = True
    
    fig.colorbar(c, ax=ax[0,:], location='right', anchor=(4, 0.3), label='Deviation of Median \n Matched B1 \n Mulitplication Factor \n from Ground Truth (%)')
    
    
    ###
    ###     PLOTTING BLOOD PRECISION
    ###
    
    for noise_plot in noise_range: 
        
        ind = noise_range.index(noise_plot) 
        index = int(noise_plot*100)
    
        #data for plotting - blood matching precision
        data_prec_blood = std_rep_array_b1[:,:,index]  
        #afmhot_r
        c = ax[1,ind].pcolor(x, y, data_prec_blood, cmap='afmhot_r', vmin=z_min_prec_b1, vmax=z_max_prec_b1, shading = 'auto')
        
        #ax[1,ind].set(xlabel='Ground Truth Blood Volume (%)', ylabel='Ground Truth Residence Time (ms)')
        #ax[1,ind].set_title('n =' + str(noise_plot))
     
    
    #plt.rcParams.update({'font.size': 48})
    #fig.subplots_adjust(right = 1.5)
    fig.colorbar(c, ax=ax[1,:], location='right', anchor=(4, 0.3), label='IQR of Matched \n B1 Mutliplication \n Factor (%)') 
    #fig.subplots_adjust(right = 1)
    
    #fig.tight_layout()
  
'''
fig,ax = plt.subplots(1,1)
signal1 = np.load('/Users/emmathomson/Desktop/Local/Dictionaries/Dictionary2000fastTE2im/echo_400_1500_200_10.0_0.9_1.npy')
signal2 = np.load('/Users/emmathomson/Desktop/Local/Dictionaries/Dictionary2000fastTE2im/echo_400_1500_1500_10.0_0.9_1.npy')
signal1 = (signal1-np.min(signal1[:,0,0]))/(np.max(signal1[:,0,0])-np.min(signal1[:,0,0]))
signal2 = (signal2-np.min(signal2[:,0,0]))/(np.max(signal2[:,0,0])-np.min(signal2[:,0,0]))


plt.plot(signal1[:,0,0], linewidth=2, label = 'tb = 200 ms')
plt.plot(signal2[:,0,0], label = 'tb = 1500 ms')

ax.set_xlabel('TR index')
ax.set_ylabel('Normalised Signal')
ax.set_title("tb differences - Normalised")
fig.set_size_inches(11,8, forward=True)
plt.ylim([0.9,0.94])
plt.xlim([0,200])
plt.legend()
plt.rcParams.update({'font.size': 24})
plt.show()
'''
    
'''-----------------------------QUIVER PLOTS----------------------------------'''

## FIND QUIVER PLOTS

# Creating arrow
xx = np.arange(1, 11, 1)
yy = np.arange(200, 1700, 100)
X, Y = np.meshgrid(xx, yy)

wid = 0.015
sca = 2

'''
    PLOT AS SINGLE FIGURE
'''

###
###     PLOTTING BLOOD ACCURACY
###

#ax = plt.subplot2grid((80,32), (0,0))
fig, ax = plt.subplots(1,4, constrained_layout=True)
fig.set_size_inches(85,26, forward=True)
plt.rcParams.update({'font.size': 90})
plt.rcParams['xtick.labelsize'] =90
plt.rcParams['ytick.labelsize'] = 90
#gs = gridspec.GridSpec(, width_ratios=[[1, 1, 2], [1, 1, 2], [1, 1, 2], [1, 1, 2]]) 

#plt.subplots_adjust(top=0.9)

fig.supylabel('Ground Truth Residence Time (ms)')

fig.supxlabel('Ground Truth Blood Volume (%)')#, ylabel='Ground Truth Residence Time (ms)')


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
x_tile = np.tile(np.expand_dims(x,2),[1,1,noise])
y_tile = np.tile(np.expand_dims(y,2),[1,1,noise])

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
ax[0].errorbar(noisePlotting, dataIQR[0,1:], yerr=dataIQR[1,1:],  capsize=7, fmt="o-", c="black", label = '2D')
ax[0].errorbar(noisePlotting5d, dataIQR5d[0,1:], yerr=dataIQR5d[1,1:],  capsize=7, fmt="o-", c="blue", label = '5D')
ax[0].set_ylabel('Mean Error in Median \n Matched Value for ' r'$\nu_b$ [%]')
ax[0].plot([0,0.1], [0.5,0.5], 'r--')
#ax[0].plot([0.07,0.07], [-0.3,0.5], 'r--')
ax[0].set_ylim(-0.3,2.5)
ax[0].set_xlim(0,0.1)
ax[1].errorbar(noisePlotting, dataIQR[2,1:], yerr=dataIQR[3,1:],  capsize=7, fmt="o-", c="black", label = '2D')
ax[1].errorbar(noisePlotting5d, dataIQR5d[2,1:], yerr=dataIQR5d[3,1:],  capsize=7, fmt="o-", c="blue", label = '5D')
ax[1].set_ylabel('Mean Error in Median \n Matched Value for ' r'$\tau_b$ [ms]')
ax[1].plot([0,0.1], [50,50], 'r--')
#ax[1].plot([0.04,0.04], [-100,50], 'r--')
ax[1].set_ylim(-100,700)
ax[1].set_xlim(0,0.1)
ax[1].legend()
ax[0].legend()

