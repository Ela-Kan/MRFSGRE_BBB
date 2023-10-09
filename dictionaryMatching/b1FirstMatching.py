#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""------------------------------------------------------------------------
Standard inner product MRF matching with B1 matched, smoothed, fixed, and the 
remaining parameters rematched 

Author: Emma Thomson
Year: 2023
Institution: Centre for Medical Image Computing: University College London
Email: e.thomson.19@ucl.ac.uk
------------------------------------------------------------------------"""

''' -----------------------------PACKAGES--------------------------------- '''

import numpy as np
import os
import glob
from pydicom import dcmread
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt   
import skimage.io
import skimage.color
import skimage.filters
import nibabel as nib
from skimage.transform import resize
from scipy import interpolate
from scipy import signal
import cv2
import warnings
warnings.filterwarnings("ignore")

#go up a folder
os.chdir("..")
print(os.getcwd())

plt.rcParams['font.family'] = 'Times'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 12

import time

t0 = time.time()

''' -----------------------------FUNCTIONS--------------------------------- '''

def get_basis(x, y, max_order=4):
    """Return the fit basis polynomials: 1, x, x^2, ..., xy, x^2y, ... etc."""
    basis = []
    for i in range(max_order+1):
        for j in range(max_order - i +1):
            basis.append(x**j * y**i)
    return basis
    

''' -----------------------------INPUTS--------------------------------- '''

#Input volunteer number
volunteer_no = 1.1

# Dictionary folder
dictfolder = 'WEXandBVfreeNew'  #'SliceProfileNew' #'WEXandBVfreeNew' #'Sequence' #'InversionRecovery'

#image resolutiopn  
res_x =  64; res_y = 64;
#length of acquisiton
acqlen = 2000

#Step size of the dictionary B1 parameter - for rounding 
b1_dict_step = 2


#Folder Path
# Image folder paths 
pathToFolder = ('./SampleData/Volunteer' + str(volunteer_no) + '/MRF')
bigpath = (pathToFolder + '/Images/') 

dictPath = ('./Dictionaries/Dictionary' + dictfolder + '/')

#Use the denoised data? 
denoise = False
#if you are denoising then what type of filter do you want to use 
filt = 'G' #G = gaussian #MF = median filtering
#if gaussian filter - how much smoothing do you want
sigma = 0.5 #1 

#Display masked images? 
mask_im = 'no'
#Save parameter maps? 
save_im = 'yes'
#Segement images?
seg_im = 'no'

#Number of entries in dictionary (to set array sizes)
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
    for filename in glob.glob(os.path.join(str('./SampleData/Volunteer' + str(volunteer_no) + '/Mask_Images/' + rr))):
        
        mask_load = nib.load(filename)
        mask = (np.flipud(np.array(mask_load.dataobj).T))

        mask_resized = cv2.resize(mask, dsize=(res_x, res_y), interpolation=cv2.INTER_NEAREST)
        
        #mask = mask.resize([res_x, res_y])
    
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


seg_path = './SampleData/Volunteer' + str(volunteer_no) + '/Mask_Images/Segmented/'

#The CSF seg is weird for some volunteers so generate it from 1 - (WM seg + GM seg)
segWM = (seg_path + 'T2_pve_' + str(1) + '.nii.gz')
fpath =  segWM
seg_load_WM = nib.load(fpath) 
segWM = np.array(seg_load_WM.dataobj)

segGM = (seg_path + 'T2_pve_' + str(2) + '.nii.gz')
fpath =  segGM
seg_load_GM = nib.load(fpath) 
segGM = np.array(seg_load_GM.dataobj)

segCSF = 1 - (segWM + segGM)

#if volunteer_no == 17.1:
segCSF = (np.rot90(segCSF))
segCSF= resize(segCSF, (res_x, res_y),
               anti_aliasing=False)

grey_image = segCSF #skimage.color.rgb2grey(segCSF)
histogram, bin_edges = np.histogram(grey_image, bins=256, range=(0.0, 1.0))
t = np.max(segCSF)*0.25
binary_seg_mask = segCSF > t
binary_seg_mask_CSF = binary_seg_mask.astype('uint8')

binary_mask_seg= np.float64(binary_mask-binary_seg_mask_CSF*binary_mask + (1-binary_mask))


binary_mask_seg[binary_mask_seg == 0] = np.NaN       

#Pad array to ensure that the ventricles and surrounding distortions are fully removed 

gradx = np.gradient(binary_mask_seg)[0] 
grady = np.gradient(binary_mask_seg)[1]
gradtot = abs(gradx) + abs(grady)        
gradtot[gradtot > 0] = 1
gradtot = abs(1-gradtot)

#binary_mask_seg[np.isnan(binary_mask_seg)] = 1
binary_mask_seg = abs(gradtot + binary_mask_seg)

binary_mask_seg[binary_mask_seg>2] = 0 
binary_mask_seg[binary_mask_seg == 2] = 1


''' --------------------------READ IN SIMS------------------------------ '''

print("Starting Dictionary Read in:  " + str(time.strftime('%X %x %Z')))

#Open empy arrays for dictionary signals
mean_sims = np.zeros([no_entries])

#Loading all dictionary signals into array    
with open(os.path.join(dictPath + "dictionary.txt" ), 'r') as f:
    lines = np.loadtxt(f, delimiter=",")
array = np.asarray(lines)


#Load lookup table  
with open(os.path.join(dictPath + "lookupTable.txt" ), 'r') as f:
    lines = np.loadtxt(f, delimiter=",")
lookup = np.asarray(lines)   
lookupList = lookup.T.tolist()#list(f for f in list(lines))


for ii in range(0,np.size(array,1)):
    array[:,ii] = (array[:,ii]/(np.linalg.norm(array[:,ii],2))) 

param_est = np.zeros([res_x,res_x,5])
norm_sims = np.linalg.norm(array, 2, axis=0)

''' -----------------------------MATCHING-------------------------------- '''

print("Starting Matching 1:  " + str(time.strftime('%X %x %Z')))


#store match quality per pixel 
error = np.zeros([res_x, res_y])
rss = np.zeros([res_x, res_y])

#FIX ME
data1 = data #[:,:,:1000]
array1 = array#[:1000,:]

#array = array[:-data_offset,:]
for pixel_x in range(res_x):
    for pixel_y in range(res_y):

        ind =[]
        
        #signal of a single voxel 

        normsig = data1[pixel_x,pixel_y,:] 
       
        #If signal is nan then save values as zeros 
        if (len(np.where(np.isnan(normsig) == False)[0]) == 0) or np.all(normsig == 0):
            param_est[pixel_x,pixel_y,:] =  0
        else: 
           #Find the maximum index (closes match)
           #Calculate the inner product of the two arrays (matching)
           
           dot_sum1 = np.inner(normsig,array1.T)
           max_index = np.argmax(abs(dot_sum1))
           
           error[pixel_x, pixel_y] = np.max(dot_sum1)
           rss[pixel_x, pixel_y] = np.sum(np.square(normsig - array1[:,max_index])[1:])

           #dot_sum1 = np.linalg.norm(array.T - normsig,2, axis=0)
           dot_sum1 = dot_sum1[dot_sum1 != 0]
           #max_index = np.argmin(abs(dot_sum1))
           ind.append(max_index)
       
           #Write 5D matches to array 
           t1t = lookup[0,max_index]
           t1b = lookup[1,max_index]
           res = lookup[2,max_index]
           perc = lookup[3,max_index]
           b1 = lookup[4,max_index]
           
           param_est[pixel_x,pixel_y,:] = [t1t, t1b, res, perc, b1]

'''--------------------------B1 SMOOTHING--------------------------------- '''

np.random.seed(42)

# The two-dimensional domain of the fit.
xmin, xmax, nx =0, 64, 64
ymin, ymax, ny = 0, 64, 64
x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y)

#Plot T1 of tissue
fig, ax = plt.subplots()
fig.set_size_inches(10,7, forward=True)
plt.imshow(np.squeeze(param_est[:,:,4]), cmap='Greys_r')
cbar = plt.colorbar()
cbar.set_label(r'Raw $\ B_1^+$')
plt.clim(0.7,1.2)

Z = param_est[:,:,4]*binary_mask
Zhold = Z
Zhold[Zhold < 0.7] = 0.69

# Plot the 3D figure of the fitted function and the residuals.
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Zhold, cmap='YlGnBu_r')
ax.set_zlim(0.7,1.2)
plt.show()

fit = 2*Z/2

#Segment out the csf and vessels
binary_mask_seg = np.float64(binary_mask_seg)
binary_mask_seg[binary_mask_seg == 0] = np.NaN
fit = fit*binary_mask_seg*binary_mask

fig,ax = plt.subplots()
plt.imshow(fit)          

while bool(list(fit[fit>1.12])) is True:
    fit[fit<0.7] = 0.69
    fit[fit>=1.06] = np.NaN

    x = np.arange(-fit.shape[1]/2, fit.shape[1]/2)
    y = np.arange(-fit.shape[0]/2, fit.shape[0]/2)
    #mask invalid values
    fit = np.ma.masked_invalid(fit)
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~fit.mask]
    y1 = yy[~fit.mask]
    newarr = fit[~fit.mask]
    
    fit = interpolate.griddata((x1, y1), newarr.ravel(),
                              (xx, yy),
                                 method='cubic')

    #correct any edge cases by setting to the lowest value
    fit = fit*binary_mask
    fit[fit < 0.7] = 0.7
    fit[fit > 1.2] = 1.2

fit = ndimage.gaussian_filter(fit, sigma=(1.5, 1.5), order=0)
fit  = fit*binary_mask
fit[fit<0.7] = 0.7

#Round to precision of the dictionary - in this case whole integers/10 
b1HoldMatch = np.round(fit*(100/b1_dict_step),0)/(100/b1_dict_step)

#correct any edge cases by setting to the lowest value
b1HoldMatch[b1HoldMatch < 0.7] = 0.7

#Plot T1 of tissue
fig, ax = plt.subplots()
fig.set_size_inches(10,7, forward=True)
plt.imshow(np.squeeze(b1HoldMatch), cmap='Greys_r')
cbar = plt.colorbar()
cbar.set_label(r'Smoothed $B_1^+$')
plt.clim(0.7,1.2)

#remask 
b1Matched = b1HoldMatch#*binary_mask

''' -----------------------------MATCHING 2-------------------------------- '''

print("Starting Matching 2:  " + str(time.strftime('%X %x %Z')))

get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]

#store match quality per pixel 
error = np.zeros([res_x, res_y])
rss = np.zeros([res_x, res_y])
snr = np.zeros([res_x, res_y])

#array = array[:-data_offset,:]
for pixel_x in range(res_x):
    for pixel_y in range(res_y):

        ind =[]
        
        #signal of a single voxel 

        normsig = data[pixel_x,pixel_y,:] 
        
        holdB1 = b1Matched[pixel_x,pixel_y] 
        B1location =  get_indexes(holdB1,lookup[4,:])
        arrayFixedB1 = array[:,B1location]
       
        #If signal is nan then save values as zeros 
        if (len(np.where(np.isnan(normsig) == False)[0]) == 0) or np.all(normsig == 0):
            param_est[pixel_x,pixel_y,:] =  0
        else: 
           #Find the maximum index (closes match)
           #Calculate the inner product of the two arrays (matching)
           dot_sum1 = np.inner(normsig,arrayFixedB1.T)
           max_index_B1 = np.argmax(abs(dot_sum1))
           
           error[pixel_x, pixel_y] = np.max(dot_sum1)
           rss[pixel_x, pixel_y] = np.sum(np.square(normsig - arrayFixedB1[:,max_index_B1])[1:])
           snr[pixel_x, pixel_y] = np.nanmean(normsig)/rss[pixel_x, pixel_y]
           
           max_index = B1location[max_index_B1]
           #dot_sum1 = np.linalg.norm(array.T - normsig,2, axis=0)
           dot_sum1 = dot_sum1[dot_sum1 != 0]
           #max_index = np.argmin(abs(dot_sum1))
           ind.append(max_index)
       
           #Write 5D matches to array k-
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
cbar.set_label(r'$Matched  \ v_{b}$')
plt.clim(1,10)

#Plot B1 multiplication
fig, ax = plt.subplots()
fig.set_size_inches(20,15, forward=True)
plt.imshow(np.squeeze(param_est[:,:,4]), cmap='Greys_r')
cbar = plt.colorbar()
cbar.set_label(r'$Matched \ B_1^+$')
plt.clim(0.5,1.2)


#Do you want to segment the images and print averages for each tissue type? 
if seg_im == 'yes':
    
    print("Starting Segmentation:  " + str(time.strftime('%X %x %Z')))
    
    seg_path = '/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + str(volunteer_no) + '/Mask_Images/Segmented/'
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
    
    
        gray_image = seg_resized #skimage.color.rgb2gray(seg_resized)
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
    for filename in glob.glob(os.path.join(str('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + str(volunteer_no) + '/Mask_Images/' + rr))):
        img = nib.load(filename)    
        aff = img.affine
                  
        im = param_est[:,:,0]
        im = np.flipud(im).T           
        new_image = nib.Nifti1Image(im, affine=aff)
        #rotate_img = np.rot90(new_image, 180)
        if denoise is True: 
            if filt == 'G':
                filestr = 'B1first_T1_t[ms]_' + dictfolder + '_G_S_' + str(sigma) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'B1first_T1_t[ms]_' + dictfolder + '_MF_S_3.nii.gz'
        else: 
             filestr = 'B1first_T1_t[ms]_' + dictfolder + '.nii.gz' 
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  
        
        im = param_est[:,:,1]
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        if denoise is True: 
            if filt == 'G':
                filestr = 'B1first_T1_b[ms]_' + dictfolder + '_G_S_' + str(sigma) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'B1first_T1_b[ms]_' + dictfolder + '_MF_S_3.nii.gz'
        else: 
            filestr = 'B1first_T1_b[ms]_' + dictfolder + '.nii.gz' 
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  
        
        im = param_est[:,:,2]
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        if denoise is True: 
            if filt == 'G':
                filestr = 'B1first_tau_b[ms]_' + dictfolder + '_G_S_' + str(sigma) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'B1first_tau_b[ms]_' + '_MF_S_3.nii.gz'
        else: 
            filestr = 'B1first_tau_b[ms]_' + dictfolder + '.nii.gz' 
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  
        
        im = param_est[:,:,3]
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        if denoise is True:
            if filt == 'G':
                filestr = 'B1first_v_b[%]_' + dictfolder + '_G_S_' + str(sigma) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'B1first_v_b[%]_' + dictfolder + '_MF_S_3.nii.gz'
        else: 
            filestr = 'B1first_v_b[%]_' + dictfolder + '.nii.gz' 
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr)) 
        
        im = param_est[:,:,4]
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        if denoise is True: 
            if filt == 'G':
                filestr = 'B1first_B1+_' + dictfolder + '_G_S_' + str(sigma) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'B1first_B1+_' + dictfolder + '_MF_S_3.nii.gz'
        else:
            filestr = 'B1first_B1+_' + dictfolder + '.nii.gz' 
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  
        
        im = param_est[:,:,5]
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        if denoise is True: 
            if filt == 'G':
                filestr = 'B1first_Relative M0[a.u.]_' + dictfolder + '_G_S_' + str(sigma) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'B1first_Relative M0[a.u.]_' + dictfolder + '_MF_S_3.nii.gz'
        else: 
            filestr = 'B1first_Relative M0[a.u.]_' + dictfolder + '.nii.gz' 
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  
        
        im = snr
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        if denoise is True:
            if filt == 'G':
                filestr = 'B1first_SNR_' + dictfolder + '_G_S_' + str(sigma) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'B1first_SNR_' + dictfolder + '_MF_S_3.nii.gz'
        else: 
            filestr = 'B1first_SNR_' + dictfolder + '.nii.gz' 
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  
        
        im = rss
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        if denoise is True: 
            if filt == 'G':
                filestr = 'B1first_Residual_' + dictfolder + '_G_S_' + str(sigma) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'B1first_Residual_' + dictfolder + '_MF_S_3.nii.gz' 
        else: 
            filestr = 'B1first_Residual_' + dictfolder + '.nii.gz' 
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  
        
        im = error
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        if denoise is True: 
            if filt == 'G':
                filestr = 'B1first_Relative error_' + dictfolder + '_G_S_' + str(sigma) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'B1first_Relative error_' + dictfolder + '_MF_S_3.nii.gz'
        else:
            filestr = 'B1first_Relative error_' + dictfolder + '.nii.gz' 
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  


#Stop timer and print                                                    
t1 = time.time()
total = t1-t0
print("Total Time:    " +  str(total) + " seconds")  

           