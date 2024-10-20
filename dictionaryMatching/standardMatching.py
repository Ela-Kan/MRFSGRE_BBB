#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""------------------------------------------------------------------------
Standard inner product MRF matching:
    - Matching is parallized 
    - Brain masking and segmentation of CSF partial voxels


Author: Emma Thomson
Year: 2022
Institution: Centre for Medical Image Computing: University College London
Email: e.thomson.19@ucl.ac.uk
------------------------------------------------------------------------"""

''' -----------------------------PACKAGES--------------------------------- '''

import numpy as np
import os
import glob
import matplotlib.pyplot as plt   
import skimage.io
import skimage.color
import skimage.filters
import nibabel as nib
from skimage.transform import resize
import cv2

plt.style.use('bmh')
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 12

import time 

t0 = time.time()

import warnings
warnings.filterwarnings("ignore")

#set cwd to the code folder

os.chdir("/Users/ela/Documents/PhD/code/")

''' -----------------------------INPUTS--------------------------------- '''

#Input volunteer number
volunteer_no = 1.1

# Dictionary folder
dictfolder = '2Dsim'  #'SliceProfileNew' #'WEXandBVfreeNew' #'Sequence' #'InversionRecovery'

#image resolutiopn  
res_x =  64; res_y = 64;
#length of acquisiton
acqlen = 2000

#Folder Path

# Image folder paths 
print('cwd')
print(os.getcwd())
pathToFolder = ('./MRFSGRE_BBB/SampleData/Volunteer' + str(volunteer_no) + '/MRF')
bigpath = (pathToFolder + '/Images/') 
dictPath = ('./MRFSGRE_BBB/dictionaries/Dictionary' + dictfolder + '/')

#Use the denoised data? 
denoise = False
#if you are denoising then what type of filter do you want to use 
filt = 'G' #G = gaussian #MF = median filtering
#if gaussian filter - how much smoothing do you want
sigma = 0.5 #1 

#Display masked images? 
mask_im = 'yes'
#Save parameter maps? 
save_im = 'yes'
#Segement images?
seg_im = 'no'

#Number of entries in dictionary (to set array sizes)
print(os.listdir(os.getcwd()))
no_entries = np.size(os.listdir(dictPath))

''' ---------------------------READ IN DATA------------------------------- '''

print("Starting Data Read in:  " + str(time.strftime('%X %x %Z')))

if denoise is True: 
    if filt == 'MF':
        imName = bigpath + 'IM_0001_MF_S_3.nii'
    elif filt == 'G':
        imName = bigpath + 'IM_0001_G_S_' + str(sigma) + '.nii'
else: 
    imName = bigpath + 'IM_0001.nii.gz'
    
#Load the MRF data
data = np.fliplr(np.squeeze(np.rot90(nib.load(imName).dataobj)))

#Normalise across each time course
for xx in range(0,res_x):
    for yy in range(0,res_y):
        data[xx,yy,:] = (data[xx,yy,:])/np.linalg.norm((data[xx,yy,:]),2)

''' -----------------------MASK BRAIN-------------------------- '''

print("Starting Masking:  " + str(time.strftime('%X %x %Z')))

binary_mask = 1
if mask_im == 'yes':
    
    rr = '*brain_mask.nii.gz' 
    for filename in glob.glob(os.path.join(str('./MRFSGRE_BBB/SampleData/Volunteer' + str(volunteer_no) + '/Mask_Images/' + rr))):
         
         mask_load = nib.load(filename)
         mask = np.fliplr(np.flipud(np.array(mask_load.dataobj).T))
         
         mask_resized = cv2.resize(mask, dsize=(res_x, res_y), interpolation=cv2.INTER_NEAREST)

    
    gray_image = mask_resized #skimage.color.rgb2gray(mask_resized)
    #blurred_image = skimage.filters.gaussian(gray_image, sigma=1.0)
    histogram, bin_edges = np.histogram(gray_image, bins=256, range=(0.0, 1.0))
    t = 3e-05
    binary_mask = mask_resized > t
    binary_mask = binary_mask.astype('uint8') #int(binary_mask == 'True')
    #binary_mask = abs(np.int64(binary_mask)-1)
    
    #Tile array for the number of images
    binary_mask_tiled = np.tile(np.expand_dims(binary_mask,2),acqlen)
    
    #Multiply to mask data
    data = data * binary_mask_tiled   
 
''' -----------------------SEGMENTATION BRAIN-------------------------- '''

print("Starting Segmentation:  " + str(time.strftime('%X %x %Z')))

if seg_im == 'yes':
    
    seg_path = './MRFSGRE_BBB/SampleData/Volunteer' + str(volunteer_no) + '/Mask_Images/Segmented/'
    
    segCSF = 0
    
    #Remove any CSF component
    segWM = (seg_path + 'T2_pve_' + str(segCSF) + '.nii.gz')
    fpath =  segWM
    seg_load_WM = nib.load(fpath) 
    segWM = np.array(seg_load_WM.dataobj)
    segWM = np.fliplr(np.rot90(segWM))
    segWM = resize(segWM, (res_x, res_y),
                   anti_aliasing=False)
    gray_image = skimage.color.rgb2gray(segWM)
    histogram, bin_edges = np.histogram(gray_image, bins=256, range=(0.0, 1.0))
    t = np.min(segWM) 
    binary_seg_mask = segWM > t
    binary_seg_mask_WM = binary_seg_mask.astype('uint8')
    
    binary_mask= binary_mask-binary_seg_mask_WM*binary_mask
    binary_mask_tiled = np.tile(np.expand_dims(binary_mask,2),acqlen)
    #Multiply to mask data
    data = data * binary_mask_tiled    

''' --------------------------READ IN SIMS------------------------------ '''

print("Starting Dictionary Read in:  " + str(time.strftime('%X %x %Z')))

#Open empy arrays for dictionary signals
mean_sims = np.zeros([no_entries])

#Loading all dictionary signals into array    
with open(os.path.join(dictPath + "dictionary.txt" ), 'r') as f:
    lines = np.loadtxt(f, delimiter=",")
array = np.asarray(lines).T


#Load lookup table  
with open(os.path.join(dictPath + "lookupTable.txt" ), 'r') as f: 
    lines = np.loadtxt(f, delimiter=",")
lookup = np.asarray(lines).T
lookupList = lookup.T.tolist()

norm_sims = np.linalg.norm(array, 2)

#Normalise dictionary signal - normalise over signal
#For scaling purposes
for ii in range(0,np.size(array,1)):
    array[:,ii] = (array[:,ii]/(np.linalg.norm(array[:,ii],2))) 

param_est = np.zeros([res_x,res_x,5]) # estimating 5 parameters

''' -----------------------------MATCHING-------------------------------- '''

print("Starting Matching:  " + str(time.strftime('%X %x %Z')))

#store match quality per pixel 
snr = np.zeros([res_x, res_y])
rss = np.zeros([res_x, res_y])

for pixel_x in range(res_x):
    for pixel_y in range(res_y):

        ind =[]
        
        #signal of a single voxel 
        normsig = data[pixel_x,pixel_y,:] 
       
        #If signal is nan then save values as zeros 
        if (len(np.where(np.isnan(normsig) == False)[0]) == 0) or np.all(normsig == 0):
            param_est[pixel_x,pixel_y,:] =  0
        else: 
           #Find the maximum index (closes match)
           #Calculate the inner product of the two arrays (matching)
           dot_sum1 = np.inner(normsig,array.T)
            
           max_index = np.argmax(abs(dot_sum1)/norm_sims)

           rss[pixel_x, pixel_y] =np.sum(np.square(normsig - array[:,max_index]))
           snr[pixel_x, pixel_y] = np.mean(normsig)/rss[pixel_x, pixel_y]
           
           dot_sum1 = dot_sum1[dot_sum1 != 0]
           ind.append(max_index)
       
           #Write 5D matches to array
           t1t = lookup[0,max_index]
           t1b = lookup[1,max_index]
           res = lookup[2,max_index]
           perc = lookup[3,max_index]
           b1 = lookup[4,max_index]

           param_est[pixel_x,pixel_y,:] = [t1t, t1b, res, perc, b1]

''' ----------------------------PLOTTING--------------------------------- ''' 
#Plot 5 parameter maps

#Plot T1 of tissue
fig, ax = plt.subplots()
fig.set_size_inches(20,15, forward=True)
plt.imshow(np.squeeze(param_est[:,:,0]), cmap='Greys_r')
cbar = plt.colorbar()
cbar.set_label(r'$Matched  \ T_{1,t}$ [ms]')
plt.clim(400,2000)

#Plot T1 of blood
fig, ax = plt.subplots()
fig.set_size_inches(20,15, forward=True)
plt.imshow(np.squeeze(param_est[:,:,1]), cmap='Greys_r')
cbar = plt.colorbar()
cbar.set_label(r'$Matched  \ T_{1,b}$ [ms]')
plt.clim(1500,2000)

#Plot residence time 
fig, ax = plt.subplots()
fig.set_size_inches(20,15, forward=True)
plt.imshow(np.squeeze(param_est[:,:,2]), cmap='Greys_r')
cbar = plt.colorbar()
cbar.set_label(r'$Matched  \ \tau_{b}$')
plt.clim(200,1600)

#Plot blood volume 
fig, ax = plt.subplots()
fig.set_size_inches(20,15, forward=True)
plt.imshow(np.squeeze(param_est[:,:,3]), cmap='Greys_r')
cbar = plt.colorbar()
cbar.set_label(r'$Matched  \ \nu_{b}$')
plt.clim(1,10)

#Plot B1 multiplication
fig, ax = plt.subplots()
fig.set_size_inches(20,15, forward=True)
plt.imshow(np.squeeze(param_est[:,:,4]), cmap='Greys_r')
cbar = plt.colorbar()
cbar.set_label(r'$Matched \ B_1^+$')
plt.clim(0.5,1.2)

snr = np.fliplr(snr)
fig, ax = plt.subplots()
fig.set_size_inches(20,15, forward=True)
plt.imshow(snr, cmap='jet', vmin = 0, vmax=50)
cbar = plt.colorbar()
cbar.set_label('SNR')

#Do you want to segment the images and print averages for each tissue type? 
if seg_im == 'yes':
    
    print("Starting Segmentation:  " + str(time.strftime('%X %x %Z')))
    
    seg_path = './MRFSGRE_BBB/SampleData/Volunteer' + str(volunteer_no) + '/Mask_Images/Segmented/'
    segnames = os.listdir(seg_path)
    #Only reading in image files
    segString = str(res_x) + "_pve"
    segnames = [ x for x in segnames if segString in x ]
    segnames.sort()
    
    quant_est = np.zeros([5,2,len(segnames)])
    #store size of each segmented tissue for stats
    segment_size = np.zeros([5,len(segnames)])
    for i in range(0,len(segnames)):
        ii = segnames[i]
        fpath = seg_path + ii
        seg_load = nib.load(fpath) 
    
        seg = np.array(seg_load.dataobj)
        
        seg_resized =  seg #resize(np.flipud(seg), (mask.shape[0] // 5.83, mask.shape[1] // 5.83),
                       #anti_aliasing=False)
    
    
        gray_image = skimage.color.rgb2gray(seg_resized)
        histogram, bin_edges = np.histogram(gray_image, bins=256, range=(0.0, 1.0))
        t = np.max(seg_resized)  - np.max(seg_resized)/100
        binary_seg_mask = seg_resized > t
        binary_seg_mask = binary_seg_mask.astype('uint8') #int(binary_mask == 'True')
        
        for param in range(0,5):
            seg_im = np.squeeze(binary_mask*binary_seg_mask*param_est[:,:,param])
            seg_array = seg_im[seg_im != 0]
            segment_size[param,i] = len(seg_array) 
            mean_quant = np.mean(seg_array); std_quant = np.std(seg_array)
            quant_est[param,:,i] = [mean_quant, std_quant]

    
#Do you want to save the parameter maps 
if save_im == 'yes': 
    
    rr = '*T2.nii.gz' 
    print(glob.glob(os.path.join(str('./MRFSGRE_BBB/SampleData/Volunteer' + str(volunteer_no) + '/Mask_Images/' + rr))))
    for filename in glob.glob(os.path.join(str('./MRFSGRE_BBB/SampleData/Volunteer' + str(volunteer_no) + '/Mask_Images/' + rr))):
        
        img = nib.load(filename)    
        aff = img.affine
                  
        im = param_est[:,:,0]
        im = np.flipud(im).T           
        new_image = nib.Nifti1Image(im, affine=aff)
        #rotate_img = np.rot90(new_image, 180)
        if denoise is True: 
            if filt == 'G':
                filestr = 'T1_t[ms]_' + dictfolder + '_G_S_' + str(sigma) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'T1_t[ms]_' + dictfolder + '_MF_S_3.nii.gz'
        else: 
             filestr = 'T1_t[ms]_' + dictfolder + '.nii.gz' 
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  
        
        im = param_est[:,:,1]
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        if denoise is True: 
            if filt == 'G':
                filestr = 'T1_b[ms]_' + dictfolder + '_G_S_' + str(sigma) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'T1_b[ms]_' + dictfolder + '_MF_S_3.nii.gz'
        else: 
            filestr = 'T1_b[ms]_' + dictfolder + '.nii.gz' 
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  
        
        im = param_est[:,:,2]
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        if denoise is True: 
            if filt == 'G':
                filestr = 'tau_b[ms]_' + dictfolder + '_G_S_' + str(sigma) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'tau_b[ms]_' + '_MF_S_3.nii.gz'
        else: 
            filestr = 'tau_b[ms]_' + dictfolder + '.nii.gz' 
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  
        
        im = param_est[:,:,3]
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        if denoise is True:
            if filt == 'G':
                filestr = 'v_b[%]_' + dictfolder + '_G_S_' + str(sigma) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'v_b[%]_' + dictfolder + '_MF_S_3.nii.gz'
        else: 
            filestr = 'v_b[%]_' + dictfolder + '.nii.gz' 
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr)) 
        
        im = param_est[:,:,4]
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        if denoise is True: 
            if filt == 'G':
                filestr = 'B1+_' + dictfolder + '_G_S_' + str(sigma) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'B1+_' + dictfolder + '_MF_S_3.nii.gz'
        else:
            filestr = 'B1+_' + dictfolder + '.nii.gz' 
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  

        im = snr
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        if denoise is True:
            if filt == 'G':
                filestr = 'SNR_' + dictfolder + '_G_S_' + str(sigma) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'SNR_' + dictfolder + '_MF_S_3.nii.gz'
        else: 
            filestr = 'SNR_' + dictfolder + '.nii.gz' 
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  
        
        im = rss
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        if denoise is True: 
            if filt == 'G':
                filestr = 'Residual_' + dictfolder + '_G_S_' + str(sigma) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'Residual_' + dictfolder + '_MF_S_3.nii.gz' 
        else: 
            filestr = 'Residual_' + dictfolder + '.nii.gz' 
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  

#Stop timer and print                                                    
t1 = time.time()
total = t1-t0
print("Total Time:    " +  str(total) + " seconds")  
