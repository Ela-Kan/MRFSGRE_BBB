#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""------------------------------------------------------------------------

Parcellation of maps + RSS map
    - Can choose what gets parcellized (idk if thats a word)
    - Brain masking and segmentation of CSF partial voxels


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
import matplotlib.pyplot as plt 
import matplotlib.lines as lines 
import matplotlib as mpl    
import skimage.io
import skimage.color
import skimage.filters
import nibabel as nib
from skimage.transform import resize
import shutup
shutup.please()
import warnings
warnings.filterwarnings("ignore")
import SimpleITK as sitk
import cv2
import scipy
import csv
from scipy.signal import wiener
import scipy.ndimage as ndimage
#import fsleyes


#Set plotting style 
plt.style.use('dark_background')
plt.rcParams.update({'font.size': 50})
plt.rcParams['font.family'] = 'Times'


import fsleyes

#os.chdir('/opt/anaconda3/lib/python3.8/site-packages/fsleyes/assets/colourmaps/brain_colours/')
fsleyes.colourmaps.init()

import cmcrameri.cm as cmc
#from cmcrameri import show_cmaps
#show_cmaps()
#import palettable.scientific.sequential as pm #import PuBu_9_r as colour
#import palettable.cartocolors.sequential as pcs2 #import PuBu_9_r as colour
colourHold =  cmc.devon#colour.mpl_colormap
colour_map1 = colourHold#fsleyes.colourmaps.getColourMap('brain_colours_6bluegrn_iso')
colour_map2 = colourHold
colour_map3 = colourHold
colour_map4 = colourHold
colour_map5 = colourHold
colour_map6 =  cmc.lajolla_r #pm.LaJolla_20_r.mpl_colormap

import time 

t0 = time.time()

''' -----------------------------INPUTS--------------------------------- '''

# What format are the images in: Classic or Enhanced DICOM?
imFormat = 'E' #'C' #'E'

#Input volunteer number
volunteer_no = 30.2
#Other imaging factors
Regime = 'Red' #Ext
acqlen = 2000
number_of_readouts = 4
TE = '2_LARGE_IRFAT'
readout = "VDS" #'VDS' 


## TO DO: Add additional identifier here for  name of folder

#number of regions 
par = 3000
sli = int(138/2)

#sometimes the maps are the wrong orientation. idk why
flip_maps = False
if volunteer_no == 17.1 or volunteer_no == 18 or volunteer_no == 21:
    flip_maps = False

#is b1 matched first? 
b1first = True
fixedT1b = False 
denoise = False
filt = 'G'
dim = 'S'
#for gaussian filter
sig = 0.5
#for median filter 
step_size = 5 #5


#what is being parcellized 
atlasLoad = True
noT1 = False

if noT1 is True:
    atlasLoad = False

if atlasLoad is True: 
    t1t = True
    t1b = True 
    vb = True
    taub = True
    b1plus = False
    rssplus = False
else: 
    t1t = False
    t1b = False 
    vb = False
    taub = False
    b1plus = False
    rssplus = False

lowBounds = False
ica = False
if ica is True:
    b1first = False
noOfComponents = 7
hydra = False

# SNR fit failure 
snrFF = True 
#criteria 
snrFFcriteria = 3.2

#image resolution 
res_x =  64; res_y = 64;

#Sometimes there is a corrupt section at the front of the brain (sinus artefact)
#mask for the numerical calculations 
#not masking 
sinusMask = np.ones([res_x, res_y])
#if want to mask sinsuses 
#sinusMask[0:12, :] = 0


#Folder Path
# Image folder paths 
pathToFolder = ('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + 
    str(volunteer_no) + '/' + str(Regime) + str(acqlen) + '_' + str(readout) +
    str(number_of_readouts) + '_TE' + str(TE))
bigpathIm = (pathToFolder + '/DICOM/') 


# Dictionary folder
dictfolder = 'WEXandBVfreeNew' #'WEXandBVfreeNew' #'SliceProfileNew' #'SliceProfile' #'Sequence' #'InversionRecovery'

dictPath = ('/Users/emmathomson/Desktop/Local/Dictionaries/Dictionary' + dictfolder + '/')

#average or rematch 
rematch = False

plotVert = True


#if you want just subcortical atlas then 1, if both cortical and subcortical then 2
atlasNo = 1

#Type of normalisation performed 
#Set to L2-norm (standard)
norm_type = 2 #1 #2
norm_technique =  "scaling" # "ma_right" #"ma_wrong" #"fabian" #"qiane" #"relative" #"wrong"


#What images do you want to load 
#Should normally be set to 'M'
imagetype = 'M'  #'I' #'R'  #'M' #'SUM' 

#Display masked images? 
mask_im = 'yes'
#Save parameter maps? 
save_im = 'yes'
#Segement images?
seg_im = 'no'
#bias correction? - DONT THINK THIS IS GOOD
bias_corr = 'no'

#Number of FA is less than acquisition length because we want to get rid of initial values 
#initial_bin = 1
#no_of_FA = acqlen-initial_bin
#data_offset = 1
#Number of entries in dictionary (to set array sizes)

#Folder Path
# Image folder paths 
pathToFolder = ('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + 
    str(volunteer_no) + '/' + str(Regime) + str(acqlen) + '_' + str(readout) +
    str(number_of_readouts) + '_TE' + str(TE))
if ica is True:
    bigpath = (pathToFolder + '/Maps_ICA/') 
elif hydra is True:
    bigpath = (pathToFolder + '/Maps_HYDRA/') 
else:
    bigpath = (pathToFolder + '/Maps/') 
    
if fixedT1b is True and b1first is True:
    bigpath = (bigpath + 'B1firstFixedT1b_')
elif b1first is True and fixedT1b is False:
    bigpath = (bigpath + 'B1first_')
elif fixedT1b is True and b1first is False:
    bigpath = (bigpath + 'FixedT1b_')
    

'''--------------------------------READ IN MAPS----------------------------'''

print("Starting Read in:  " + str(time.strftime('%X %x %Z')))
   
#Load in maps
if hydra is True:
    t1tMaps = np.rot90(nib.load((bigpath + 'T1_t[ms].nii.gz'))._dataobj)
    t1bMaps = np.rot90(nib.load((bigpath + 'T1_b[ms].nii.gz'))._dataobj)
    taubMaps = np.rot90(nib.load((bigpath + 'tau_b[ms].nii.gz'))._dataobj)
    vbMaps = np.rot90(nib.load((bigpath + 'v_b[%].nii.gz'))._dataobj)
    b1Maps = np.rot90(nib.load((bigpath + 'B1+.nii.gz'))._dataobj)
    rssMaps = np.rot90(nib.load((bigpath + 'Residual_' + dictfolder + '.nii.gz'))._dataobj)
elif denoise is True: 
    if filt == 'G':
        t1tMaps = np.rot90(nib.load((bigpath + 'T1_t[ms]_' + dictfolder + '_G_' + dim + '_' + str(sig) +'.nii.gz'))._dataobj)
        t1bMaps = np.rot90(nib.load((bigpath + 'T1_b[ms]_' + dictfolder + '_G_' + dim + '_' + str(sig) +'.nii.gz'))._dataobj)
        taubMaps = np.rot90(nib.load((bigpath + 'tau_b[ms]_' + dictfolder + '_G_' + dim + '_' + str(sig) +'.nii.gz'))._dataobj)
        vbMaps = np.rot90(nib.load((bigpath + 'v_b[%]_' + dictfolder + '_G_' + dim + '_' + str(sig) +'.nii.gz'))._dataobj)
        b1Maps = np.rot90(nib.load((bigpath + 'B1+_' + dictfolder + '_G_' + dim + '_' + str(sig) +'.nii.gz'))._dataobj)
        snr = np.rot90(nib.load((bigpath + 'SNR_' + dictfolder + '_G_' + dim + '_' + str(sig) +'.nii.gz'))._dataobj)
        rssMaps = np.rot90(nib.load(bigpath + 'Residual_' + dictfolder + '_G_' + dim + '_' + str(sig) +'.nii.gz')._dataobj)
    elif filt == 'MF':
        t1tMaps = np.rot90(nib.load((bigpath + 'T1_t[ms]_' + dictfolder + '_MF_' + dim + '_' + str(step_size) +'.nii.gz'))._dataobj)
        t1bMaps = np.rot90(nib.load((bigpath + 'T1_b[ms]_' + dictfolder + '_MF_' + dim + '_' + str(step_size) +'.nii.gz'))._dataobj)
        taubMaps = np.rot90(nib.load((bigpath + 'tau_b[ms]_' + dictfolder + '_MF_' + dim + '_' + str(step_size) +'.nii.gz'))._dataobj)
        vbMaps = np.rot90(nib.load((bigpath + 'v_b[%]_' + dictfolder + '_MF_' + dim + '_' + str(step_size) +'.nii.gz'))._dataobj)
        b1Maps = np.rot90(nib.load((bigpath + 'B1+_' + dictfolder + '_MF_' + dim + '_' + str(step_size) +'.nii.gz'))._dataobj)
        snr = np.rot90(nib.load((bigpath + 'SNR_' + dictfolder + '_MF_' + dim + '_' + str(step_size) +'.nii.gz'))._dataobj)
        rssMaps = np.rot90(nib.load(bigpath + 'Residual_' + dictfolder + '_MF_' + dim + '_' + str(step_size) +'.nii.gz')._dataobj)
elif ica is True:
    t1tMaps = np.rot90(nib.load((bigpath + 'T1_t[ms]_components' + str(noOfComponents) + dictfolder + '.nii.gz'))._dataobj)
    t1bMaps = np.rot90(nib.load((bigpath + 'T1_b[ms]_components' + str(noOfComponents) + dictfolder + '.nii.gz'))._dataobj)
    taubMaps = np.rot90(nib.load((bigpath + 'tau_b[ms]_components' + str(noOfComponents) + dictfolder + '.nii.gz'))._dataobj)
    vbMaps = np.rot90(nib.load((bigpath + 'v_b[%]_components' + str(noOfComponents) + dictfolder + '.nii.gz'))._dataobj)
    b1Maps = np.rot90(nib.load((bigpath + 'B1+_components' + str(noOfComponents) + dictfolder + '.nii.gz'))._dataobj)
    snr = np.rot90(nib.load((bigpath + 'SNR_components' + str(noOfComponents) + dictfolder + '.nii.gz'))._dataobj)
    rssMaps = np.rot90(nib.load(bigpath + 'Residual_components' + str(noOfComponents) + dictfolder + '.nii.gz')._dataobj)
else:
    t1tMaps = np.rot90(nib.load((bigpath + 'T1_t[ms]_' + dictfolder + '.nii.gz'))._dataobj)
    t1bMaps = np.rot90(nib.load((bigpath + 'T1_b[ms]_' + dictfolder + '.nii.gz'))._dataobj)
    taubMaps = np.rot90(nib.load((bigpath + 'tau_b[ms]_' + dictfolder + '.nii.gz'))._dataobj)
    vbMaps = np.rot90(nib.load((bigpath + 'v_b[%]_' + dictfolder + '.nii.gz'))._dataobj)
    b1Maps = np.rot90(nib.load((bigpath + 'B1+_' + dictfolder + '.nii.gz'))._dataobj)
    snr = np.rot90(nib.load((bigpath + 'SNR_' + dictfolder + '.nii.gz'))._dataobj)
    rssMaps = np.rot90(nib.load(bigpath + 'Residual_' + dictfolder + '.nii.gz')._dataobj)
'''
m0Maps = np.rot90(nib.load((bigpath + 'Relative M0[a.u.].nii.gz'))._data)
errorMaps = np.rot90(nib.load((bigpath + 'Relative error.nii.gz'))._data)
'''

if flip_maps is True: 
    t1tMaps = np.fliplr(t1tMaps)
    t1bMaps = np.fliplr(t1bMaps)
    taubMaps = np.fliplr(taubMaps)
    vbMaps = np.fliplr(vbMaps)
    b1Maps = np.fliplr(b1Maps)
    rssMaps = np.fliplr(rssMaps)
    snr = np.fliplr(snr)
 
'''--------------------------------LOAD T2 IMAGE----------------------------'''

pathtoT2 = ('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + 
    str(volunteer_no) +'/Mask_Images/T2_reg.nii.gz')

t2bigimage = (np.rot90(np.squeeze((nib.load(pathtoT2)._dataobj))))

t2image = cv2.resize(t2bigimage, dsize=(res_x, res_y), interpolation=cv2.INTER_NEAREST)


#pad T2 image for plotting  
t2pad =  np.pad(t2bigimage, pad_width=((40, 0), (10, 0)))
#t2pad = np.pad(t2image, pad_width=((5, 5), (20, 0)))
if volunteer_no == 17.1 or volunteer_no == 18 or volunteer_no == 21: 
    t2pad =  np.pad(t2bigimage, pad_width=((70, 0), (10, 0)))


'''--------------------------------RESIZE QMAPS----------------------------'''

x,y = np.shape(t2image)

t1tMaps_resized = resize(t1tMaps, (x, y), anti_aliasing=True)
t1bMaps_resized = resize(t1bMaps, (x, y), anti_aliasing=True)
taubMaps_resized = resize(taubMaps, (x, y), anti_aliasing=True)
vbMaps_resized = resize(vbMaps, (x, y), anti_aliasing=True)
b1Maps_resized = resize(b1Maps, (x, y), anti_aliasing=True)
rssMaps_resized = resize(rssMaps, (x, y), anti_aliasing=True)
snr_resized = resize(snr, (x, y), anti_aliasing=True)

''' -----------------------MASK BRAIN-------------------------- '''
binary_mask = 1 
mask = 1
if mask_im == 'yes':
    rr = '*brain_mask.nii.gz' 
    for filename in glob.glob(os.path.join(str('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + str(volunteer_no) + '/Mask_Images/' + rr))):
        
        mask_load = nib.load(filename)
        mask = (np.flipud(np.array(mask_load.dataobj).T))
        
        mask_resized = cv2.resize(mask, dsize=(res_x, res_y), interpolation=cv2.INTER_NEAREST)
        #mask = mask.resize([res_x, res_y])
        
        try:
            grey_image = skimage.color.rgb2grey(mask_resized)
        except: 
            grey_image = mask_resized
        #blurred_image = skimage.filters.gaussian(grey_image, sigma=1.0)
        histogram, bin_edges = np.histogram(grey_image, bins=256, range=(0.0, 1.0))
        t = 3e-05
        binary_mask = mask_resized > t
        binary_mask_csf = binary_mask.astype('uint8')#int(binary_mask == 'True')

if seg_im == 'yes':  
    seg_path = '/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + str(volunteer_no) + '/Mask_Images/Segmented/'
    
    segCSF = 0
    if volunteer_no ==6: 
        segCSF = 5
    elif volunteer_no ==3: 
        segCSF = 2
    elif volunteer_no ==8: 
        segCSF = 5
    
    #Remove any CSF component
    segCSF = (seg_path + 'T2_pve_' + str(segCSF) + '.nii.gz')
    fpath =  segCSF
    seg_load_CSF = nib.load(fpath) 
    segCSF = np.array(seg_load_CSF.dataobj)
    #if volunteer_no == 17.1:
    segCSF = (np.rot90(segCSF))
    segCSF= resize(segCSF, (res_x, res_y),
                   anti_aliasing=False)
    try:
        grey_image = skimage.color.rgb2grey(segCSF)
    except:
        grey_image = segCSF
    histogram, bin_edges = np.histogram(grey_image, bins=256, range=(0.0, 1.0))
    t = np.max(segCSF)*0.1
    binary_seg_mask = segCSF > t
    binary_seg_mask_CSF = binary_seg_mask.astype('uint8')
    
    binary_mask= binary_mask-binary_seg_mask_CSF*binary_mask
    
''' ----------------------------SNR CRITERIA --------------------------------- ''' 

if snrFF is True: 
    if flip_maps is True:
        snr = np.fliplr(snr)
    snrMask = np.zeros(np.shape(snr))
    snrMask[snr>snrFFcriteria] = 1 
    #binary_mask = binary_mask * snrMask
    
    fig, ax = plt.subplots()
    plt.imshow(t2image,cmap='gray')
    
    snrmaskplot = snrMask*binary_mask
    snrmask = (1-snrMask)*binary_mask_csf
    snrmask[snrmask==0] = np.NaN
    
    #pltatlas[np.isnan(pltatlas)] = 0
    plt.imshow(snrmask, alpha=1, cmap = 'Set1')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
else:
    snrmaskplot = 1
    
    
t1tMaps_resized = t1tMaps_resized*binary_mask
t1bMaps_resized = t1bMaps_resized*binary_mask
taubMaps_resized = taubMaps_resized*binary_mask
vbMaps_resized = vbMaps_resized*binary_mask
b1Maps_resized = b1Maps_resized*binary_mask
rssMaps_resized = rssMaps_resized*binary_mask
snr_resized = snr_resized*binary_mask

#nan all zero values
t1tMaps_resized[t1tMaps_resized==0] = np.nan
t1bMaps_resized[t1bMaps_resized==0] = np.nan
taubMaps_resized[taubMaps_resized==0] = np.nan
vbMaps_resized[vbMaps_resized ==0] = np.nan
b1Maps_resized[b1Maps_resized==0] = np.nan
rssMaps_resized[rssMaps_resized==0] = np.nan
snr_resized[snr_resized==0] = np.nan

''' --------------------------READ IN ATLAS------------------------------ '''  

if atlasLoad is True: 
    print("Starting Atlas Read in:  " + str(time.strftime('%X %x %Z'))) 
    
    
    #subAtlasFile = ('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + 
    #    str(volunteer_no) +'/Anatomy_Seg/outputAtlas.nii.gz')
    subAtlasFile = ('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + 
        str(volunteer_no) +'/Anatomy_Seg/outputAtlas.nii.gz')
    
    subAtlasLoad = nib.load(subAtlasFile)
    subAtlas = np.array(subAtlasLoad.dataobj)
    
    atlasSliceSub = np.round((np.flipud(subAtlas[:,:,sli].T)),0)

    '''
    atlasSliceSub[atlasSliceSub == 0] = np.nan
    atlasSliceSub = ndimage.gaussian_filter(atlasSliceSub, sigma=(0.25,0.25), order=0)
    atlasSliceSub = scipy.signal.medfilt(atlasSliceSub, kernel_size=9)
    atlasSliceSub = wiener(atlasSliceSub, (4, 4)) 
    
    atlasSliceSub[atlasSliceSub>2000] = 42
    atlasSliceSub[atlasSliceSub>1000] = 3
    atlasSliceSub[atlasSliceSub==24] = 0
    '''
    
    if atlasNo == 2:
        
        #atlasSliceSub = scipy.signal.medfilt(atlasSliceSub, kernel_size=7)
        atlasSliceSub[atlasSliceSub == 0] = np.nan
        #Remove grey matter segs to overlay with cortical segs 
        atlasSliceSub[atlasSliceSub==2] = 1
        atlasSliceSub[atlasSliceSub==13] = 12
        
        cortAtlasFile = ('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + 
            str(volunteer_no) +'/Anatomy_Seg/outputAtlas-cort.nii.gz')
        
        cortAtlasLoad = nib.load(cortAtlasFile)
        cortAtlas = np.array(cortAtlasLoad.dataobj)
        atlasSliceCort = np.round((np.flipud(cortAtlas[:,:,sli].T)),0)
        #atlasSliceCort = scipy.signal.medfilt(atlasSliceCort, kernel_size=5)
        #To remove overlap with atlasSliceSub segs
        atlasSliceCort += 20
        
        '''
        cortAtlasFile = ('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + 
            str(volunteer_no) +'/Anatomy_Seg/outputAtlas-cort2.nii.gz')
        
        cortAtlasLoad = nib.load(cortAtlasFile)
        cortAtlas = np.array(cortAtlasLoad.dataobj)
        atlasSliceCort = np.round((np.flipud(cortAtlas[:,:,sli].T)),0)
        atlasSliceCort += 20
        atlasSliceCort[atlasSliceCort == 20] = 0
        atlasSliceCort[atlasSliceCort == 0] = np.nan
        #atlasSliceCort = ndimage.gaussian_filter(atlasSliceCort, sigma=(0.25,0.25), order=0)
        #atlasSliceCort = scipy.signal.medfilt(atlasSliceCort, kernel_size=7)
        #atlasSliceCort = ndimage.gaussian_filter(atlasSliceCort, sigma=(0.1,0.1), order=0)
        #atlasSliceCort = wiener(atlasSliceCort, (4, 4)) 
        atlasSliceCort[np.isnan(atlasSliceCort)] = 0
        
        atlas = np.zeros([np.size(atlasSliceSub,0),np.size(atlasSliceSub,1)])
        for xx in range(np.size(atlasSliceSub,0)):
           for yy in range(np.size(atlasSliceSub,1)):
               if atlasSliceCort[xx,yy] == 0:
                   atlas[xx,yy] = atlasSliceSub[xx,yy]
               else:
                   atlas[xx,yy] = atlasSliceCort[xx,yy]
        '''
        atlas = atlasSliceSub + atlasSliceCort
        
    else: 
        atlas = atlasSliceSub
    
    #atlas = scipy.signal.medfilt(atlas, kernel_size=9)
    atlas_resize_s = cv2.resize(atlas, dsize=(res_x, res_y), interpolation=cv2.INTER_NEAREST)
    atlas_resize_l = cv2.resize(atlas, dsize=(np.shape(t2bigimage)), interpolation=cv2.INTER_NEAREST)
          
    
    '''--------------------------------PLOT OVERLAY----------------------------'''
    #Check to see if registration has been performed well 
    '''
    fig, ax = plt.subplots(1,par)
    fig.set_size_inches(30,5, forward=True)
    
    for f in range(par):
        regmap = np.zeros(np.shape(t2bigimage))
        regmap[atlas_resize_l == f+1] = 1
        ax[f].imshow(t2bigimage, cmap='gray',  interpolation='none') 
        ax[f].imshow(regmap, cmap='Reds',  interpolation='none', alpha = 0.4) 
    
    plt.show()
    
    fig, ax = plt.subplots(1,par)
    fig.set_size_inches(30,5, forward=True)
    
    for f in range(par):
        regmap = np.zeros([res_x, res_y])
        regmap[atlas_resize_s == f+1] = 1
        ax[f].imshow(t2image, cmap='gray',  interpolation='none') 
        ax[f].imshow(regmap*binary_mask, cmap='Reds',  interpolation='none', alpha = 0.4) 
    
    plt.show()
    '''

    fig, ax = plt.subplots()
    plt.imshow(t2bigimage,cmap='gray')
    atlasRound = np.round(atlas_resize_l,0)
    pltatlas = (atlasRound)*mask
    
    #pltatlas[np.isnan(pltatlas)] = 0
    plt.imshow(np.log(pltatlas), alpha=1, cmap = colour_map1)

''' -----------------------------REMATCH??------------------------------ '''  


if rematch is True: 
    
    ''' ---------------------DATA READ IN--------------------------- '''  
    
    print("Starting Data Read in:  " + str(time.strftime('%X %x %Z')))

    #Read in data 
    imagenames = os.listdir(bigpathIm)
    #Only reading in image files
    if imFormat == 'E':
        imagenames = [ x for x in imagenames if "IM_0001" in x ]
    else: 
        imagenames = [ x for x in imagenames if "IM" in x ]
    if (len(imagenames) == 0):
        print('This folder contains no images...')
    else: 
        #sort images into numerical (aquisition) order
        imagenames.sort()
    
        #Open empty arrays for experimental signals
        store=[]
        mean_data = np.zeros([res_x,res_y])
        data = np.zeros([res_x,res_y,acqlen])
        datad = np.zeros([res_x,res_y,acqlen])
        
        #Read in the data from each image 
        ccount  = 0
        if imFormat =='E': 
            fpath = bigpathIm + imagenames[0]
            ds = dcmread(fpath)    
            datahold = ds.pixel_array
            irsum = ds.ImageType[-2]      
            privBlock = str(ds.private_block)
            infotxt = privBlock.splitlines()
            slopesAll = [ii for ii in infotxt if "0028, 1053" in ii]
            slope = slopesAll[1::2]
            slope = [ii.split(':')[1] for ii in slope]
            cleanSlope = [ii.split("'")[1] for ii in slope]
            scale = np.float64(cleanSlope)[:acqlen]
    
            '''
            for i in range(0,acqlen):
                datad[:,:,ccount] = datahold[i,:,:]
                ccount += 1
            '''
            data = np.flipud(np.rot90(datahold[:acqlen,:,:].T))*scale
    
        else:
            store=[]
            for i in range(0,len(imagenames)):
                ii = imagenames[i]
                fpath = bigpathIm + ii
                ds = dcmread(fpath)    
                imagenum = ii.split('_')[1]
                #Check that you are reading in the correct type of image 
                #Normally 'M' - magnitude image
                #if irsum == imagetype:
                datahold = ds.pixel_array
                datad[:,:,ccount] = datahold #/np.float16(ds[0x0028, 0x1053].value)
                ccount += 1
                #Optional: extract some extra parameters from the image headers
                cardname = ds.SeriesDescription
                #creationtime = ds.InstanceCreationTime
                acqnum = ds.AcquisitionNumber
                store.append([ds[0x0028, 0x1053].value])
             #Remove initial values from data array 
            data = datad[:,:,:]*np.float16(store).T
    
    
        if norm_technique == "ma_right" or  norm_technique == "ma_wrong" or norm_technique == "relative":
            #Normalise experimental signal - normalise over signal 
            #data = -data
            for xx in range(0,res_x):
                for yy in range(0,res_y):
                    mean_data[xx,yy] = np.linalg.norm((data[xx,yy,:]),norm_type)  
                    data[xx,yy,:] = (data[xx,yy,:])/np.linalg.norm((data[xx,yy,:]),norm_type)#*scale
    
                    
        if norm_technique == "fabian" or  norm_technique == "other" or norm_technique == "wrong" :
            #Normalise experimental signal - normalise over signal 
            for xx in range(0,res_x):
                for yy in range(0,res_y):
                    mean_data[xx,yy] = np.linalg.norm((data[xx,yy,:]),norm_type) 
                    data[xx,yy,:] = (data[xx,yy,:]-np.min(data[xx,yy,:]))/(np.max(data[xx,yy,:]) - np.min(data[xx,yy,:]))*1000#*scale
       
        if norm_technique == "quiane":
            #Normalise experimental signal - normalise over signal 
            for xx in range(0,res_x):
                for yy in range(0,res_y):
                    mean_data[xx,yy] = np.linalg.norm((data[xx,yy,:]),norm_type)   
                    data[xx,yy,:] = data[xx,yy,:]/(np.max(data[xx,yy,:]))
                    
        if norm_technique == "scaling":
            #data = -data
            for xx in range(0,res_x):
                for yy in range(0,res_y):
                    mean_data[xx,yy] = np.linalg.norm((data[xx,yy,:]),norm_type)  
                    
                    q75,q25 = np.percentile(data[xx,yy,:],[75,25])
                    intr_qr = q75-q25
                    #Remove outlying spikes
                    maxdata = q75+(1.5*intr_qr)
                    mindata = q25-(1.5*intr_qr)
                    
                    #Robust scaling
                    #data[xx,yy,:] = (data[xx,yy,:])/np.linalg.norm((data[xx,yy,:]),norm_type)
                    #data[xx,yy,:] = np.exp((data[xx,yy,:]-np.min(data[xx,yy,:]))/(np.max(data[xx,yy,:]) - np.min(data[xx,yy,:])))
                    data[xx,yy,:] = ((data[xx,yy,:]-np.median(data[xx,yy,:]))/(maxdata-mindata)) #q75-q25))
                   
    data = data*np.tile(np.expand_dims(binary_mask,2),acqlen)
    ''' ---------------------DICTIONARY READ IN--------------------------- '''   
    
    print("Starting Dictionary Read in:  " + str(time.strftime('%X %x %Z')))
    fff = " /Users/emmathomson/Dropbox/Coding/BBB_MRFSGRE/dictionaryFreeMatching/TrainingData/DictionaryInversionRecovery"
    #Loading all dictionary signals into array    
    with open(os.path.join(dictPath + "dictionary.txt" ), 'r') as f:
        lines = np.loadtxt(f, delimiter=",")
    array = np.asarray(lines).T     
    #array = array.astype(int)
    
    #Load lookup table  
    with open(os.path.join(dictPath + "lookupTable.txt" ), 'r') as f:
        lines = np.loadtxt(f, delimiter=",")
    lookup = np.asarray(lines).T       
    lookupList = lookup.T.tolist()#list(f for f in list(lines))
    mean_sims = np.zeros([np.size(array,1)])
    '''
    for ii in range(0,acqlen):
        array[ii,:] = (array[ii,:]/(np.linalg.norm(~np.isnan(array[ii,:]),norm_type)))
    '''
    
    #array = array/np.expand_dims(scale,axis=1) 
    
    #Normalise dictionary signal - normalise over signal
    #For scaling purposes
    
    if norm_technique == 'scaling':
        for ii in range(0,np.size(array,1)):
            mean_sims[ii] = (np.linalg.norm(array[:,ii],norm_type))
            #array[:,ii] = (array[:,ii]- np.min(array[:,ii]))/(np.max(array[:,ii])-np.min(array[:,ii]))
            q75,q25 = np.percentile(array[:,ii],[75,25])
            intr_qr = q75-q25
            #Remove outlying spikes
            maxdata = q75+(1.5*intr_qr)
            mindata = q25-(1.5*intr_qr)
            #array[:,ii] = (array[:,ii])/10
            array[:,ii]= ((array[:,ii]-np.median(array[:,ii]))/(maxdata-mindata))#(q75-q25))
            #array[:,ii] = (array[:,ii]/(np.linalg.norm(array[:,ii],norm_type))) 
            #array[:,ii] = (array[:,ii]-np.min(array[:,ii]))/(np.max(array[:,ii]) - np.min(array[:,ii])) #* mean_data
        #arr ay = array/np.linalg.norm(data[~np.isnan(data)])
    
    if norm_technique == 'ma_right':
        
        for ii in range(0,np.size(array,1)):
            mean_sims[ii] = (np.linalg.norm(array[:,ii],norm_type))
            array[:,ii] = (array[:,ii]/(np.linalg.norm(array[:,ii],norm_type))) 
            #array[:,ii] = (array[:,ii]-np.min(array[:,ii]))/(np.max(array[:,ii]) - np.min(array[:,ii])) #* mean_data
        #arr ay = array/np.linalg.norm(data[~np.isnan(data)])
        
    #array = array/np.expand_dims(scale,axis=1)
    
    ''' ---------------------EXTRACT REGIONAL AVERAGES--------------------------- '''  
    if atlasLoad is True:    
        regSigs = np.zeros([par, 5, res_x ,res_x])
        
        print("Starting Matching:  " + str(time.strftime('%X %x %Z')))
        
        for ii in range(1,par): 
        
            atlashold = np.zeros([res_x, res_y])
            atlashold[:] = np.NaN
            atlashold[np.round(atlas_resize_s,0) == ii] = 1
            
            atlasData = data*np.tile(np.expand_dims(atlashold,2),acqlen) 
            atlasSignal = np.nanmean(np.nanmean(atlasData,0),0)
    
            #Find the maximum index (closes match)
            #Calculate the inner product of the two arrays (matching)
            dot_sum1 = np.inner(atlasSignal,array.T)
            max_index = np.argmax(abs(dot_sum1))
    
            #dot_sum1 = np.linalg.norm(array.T - normsig,2, axis=0)
            dot_sum1 = dot_sum1[dot_sum1 != 0]
            #max_index = np.argmin(abs(dot_sum1))
        
            #Write 5D matches to array k-
            t1t = lookup[0,max_index]
            t1b = lookup[1,max_index]
            res = lookup[2,max_index]
            perc = lookup[3,max_index]
            b1 = lookup[4,max_index]
         
            regSigs[ii,0,:,:] = atlashold*t1t
            regSigs[ii,1,:,:] = atlashold*t1b
            regSigs[ii,2,:,:] = atlashold*perc
            regSigs[ii,3,:,:] = atlashold*res
            regSigs[ii,4,:,:] = atlashold*b1
            
        regSigs[regSigs == 0] = np.NaN
        reg = np.nansum(regSigs,0)
        reg[reg == 0] = np.NaN
        
else:
    ''' -------------------------IF NO REMATCH-------------------------------- '''  
    if atlasLoad is True:
        ''' ---------------------EXTRACT REGIONAL AVERAGES--------------------------- '''  
        regSigs = np.zeros([par, 6, res_x ,res_x])
        matches = [['Index', 'T1t' , 'T1b', 'vb', 'taub','b1', 'RSS']]
        
        for i in range(1,par): 
            
            #ii = par+1
            atlashold = np.zeros([res_x, res_y])
            atlashold[:] = np.NaN
            atlashold[np.round(atlas_resize_s,0) == i] = 1
            
            try:
                snrmaskplot[snrmaskplot==0] = np.NaN
                snrmaskplot[np.isnan(snrmaskplot)] = 0
                snrmaskplot = abs(snrmaskplot)
            except:
                snrmaskplot = 1
            
            t1tHold = t1tMaps_resized*atlashold
            t1tHoldmean = t1tHold*snrmaskplot
            t1tHold[t1tHold>0] = np.nanmean(t1tHoldmean[t1tHoldmean>0])
            regSigs[i,0,:,:] = t1tHold
            
            t1bHold = t1bMaps_resized*atlashold
            t1bHoldmean = t1bHold*snrmaskplot
            t1bHold[t1bHold>0] = np.nanmean(t1bHoldmean[t1bHoldmean>0])
            regSigs[i,1,:,:] = t1bHold
        
            vbHold = vbMaps_resized*atlashold
            vbHoldmean = vbHold*snrmaskplot
            vbHold[vbHold>0] = np.nanmean(vbHoldmean[vbHoldmean>0])
            regSigs[i,2,:,:] = vbHold
            
            taubHold = taubMaps_resized*atlashold
            taubHoldmean = taubHold*snrmaskplot
            taubHold[taubHold>0] = np.nanmean(taubHoldmean[taubHoldmean>0])
            regSigs[i,3,:,:] = taubHold
            
            b1Hold = b1Maps_resized*atlashold
            b1Holdmean = b1Hold*snrmaskplot
            b1Hold[b1Hold>0] = np.nanmean(b1Holdmean[b1Holdmean>0])
            regSigs[i,4,:,:] = b1Hold
            
            rssHold = rssMaps_resized*atlashold
            rssHoldmean = rssHold*snrmaskplot
            rssHold[rssHold>0] = np.nanmean(rssHoldmean[rssHoldmean>0])
            regSigs[i,5,:,:] = rssHold
            
            if np.nansum(atlashold) > 0:
               matches.append([i, np.nanmean(t1tHold), np.nanmean(t1bHold), np.nanmean(vbHold), np.nanmean(taubHold),np.nanmean(b1Hold),np.nanmean(rssHold)]) 
        
        regSigs[regSigs == 0] = np.NaN
        reg = np.nansum(regSigs,0)
        reg[reg == 0] = np.NaN
        
    elif noT1 is True: 
        #2: WM, GM,
        regSigs = np.zeros([2,6, res_x ,res_x])
        matches = [['T1t' , 'T1b', 'vb', 'taub','b1', 'RSS']]
    
        #WM Component
        segWM = (seg_path + 'T2_pve_2.nii.gz')
        fpath =  segWM
        seg_load_WM = nib.load(fpath) 
        segWM = np.array(seg_load_WM.dataobj)
        segWM = (np.rot90(segWM))
        segWM= resize(segWM, (res_x, res_y),
                           anti_aliasing=False)
        try:
            grey_image = skimage.color.rgb2grey(segWM)
        except:
            grey_image = segWM
        histogram, bin_edges = np.histogram(grey_image, bins=256, range=(0.0, 1.0))
        t = np.max(segWM)*0.999
        binary_seg_mask = segWM > t
        binary_seg_mask_WM = binary_seg_mask.astype('float64')
        binary_seg_mask_WM[binary_seg_mask_WM==0] = np.NaN
        
        #GM Component
        segGM = (seg_path + 'T2_pve_1.nii.gz')
        fpath =  segGM
        seg_load_GM = nib.load(fpath) 
        segGM = np.array(seg_load_GM.dataobj)
        segGM = (np.rot90(segGM))
        segGM= resize(segGM, (res_x, res_y),
                           anti_aliasing=False)
        try:
            grey_image = skimage.color.rgb2grey(segGM)
        except:
            grey_image = segGM
        histogram, bin_edges = np.histogram(grey_image, bins=256, range=(0.0, 1.0))
        t = np.max(segGM)*0.999
        binary_seg_mask = segGM > t
        binary_seg_mask_GM = binary_seg_mask.astype('float64')
        binary_seg_mask_GM[binary_seg_mask_GM==0] = np.NaN
                    
        t1tHoldWM = t1tMaps_resized*binary_seg_mask_WM
        #t1tHoldWM[t1tHoldWM>0] = np.nanmean(t1tHoldWM)
        regSigs[0,0,:,:] = t1tHoldWM
        
        t1tHoldGM = t1tMaps_resized*binary_seg_mask_GM
        #t1tHoldGM[t1tHoldGM>0] = np.nanmean(t1tHoldGM)
        regSigs[1,0,:,:] = t1tHoldGM
        
        t1bHoldWM = t1bMaps_resized*binary_seg_mask_WM
        #t1bHoldWM[t1tHoldWM>0] = np.nanmean(t1bHoldWM)
        regSigs[0,1,:,:] = t1bHoldWM
        
        t1bHoldGM = t1bMaps_resized*binary_seg_mask_GM
        #t1bHoldGM[t1tHoldGM>0] = np.nanmean(t1bHoldGM)
        regSigs[1,1,:,:] = t1bHoldGM
        
        vbHoldWM = vbMaps_resized*binary_seg_mask_WM
        #vbHoldWM[vbHoldWM>0] = np.nanmean(vbHoldWM)
        regSigs[0,2,:,:] = vbHoldWM
        
        vbHoldGM = vbMaps_resized*binary_seg_mask_GM
        #vbHoldGM[vbHoldGM>0] = np.nanmean(vbHoldGM)
        regSigs[1,2,:,:] = vbHoldGM
            
        taubHoldWM = taubMaps_resized*binary_seg_mask_WM
        #taubHoldWM[taubHoldWM>0] = np.nanmean(taubHoldWM)
        regSigs[0,3,:,:] = taubHoldWM
        
        taubHoldGM = taubMaps_resized*binary_seg_mask_GM
        #taubHoldGM[taubHoldGM>0] = np.nanmean(taubHoldGM)
        regSigs[1,3,:,:] = taubHoldGM
            
        b1HoldWM = b1Maps_resized*binary_seg_mask_WM
        #b1HoldWM[b1HoldWM>0] = np.nanmean(b1HoldWM)
        regSigs[0,4,:,:] = b1HoldWM
        
        b1HoldGM = b1Maps_resized*binary_seg_mask_GM
        #b1HoldGM[b1HoldGM>0] = np.nanmean(b1HoldGM)
        regSigs[1,4,:,:] = b1HoldGM
            
        rssHoldWM = rssMaps_resized*binary_seg_mask_WM
        #rssHoldWM[rssHoldWM>0] = np.nanmean(rssHoldWM)
        regSigs[0,5,:,:] = rssHoldWM
        
        rssHoldGM = rssMaps_resized*binary_seg_mask_GM
        #rssHoldGM[rssHoldGM>0] = np.nanmean(rssHoldGM)
        regSigs[1,5,:,:] = rssHoldGM
                
        regSigs[regSigs == 0] = np.NaN
        wmgmmeans = np.nanmean(np.nanmean(regSigs,2),2)
        wmgmstd = np.nanstd(np.nanstd(regSigs,2),2)
        
        wmcount = np.nansum(binary_seg_mask_WM)
        gmcount = np.nansum(binary_seg_mask_GM)
        
        [tstat, t1t_pValue] = scipy.stats.ttest_ind(np.ndarray.flatten(t1tHoldWM), np.ndarray.flatten(t1tHoldGM), equal_var=False, nan_policy='omit', alternative='two-sided')
        [tstat, t1b_pValue] = scipy.stats.ttest_ind(np.ndarray.flatten(t1bHoldWM), np.ndarray.flatten(t1bHoldGM), equal_var=False, nan_policy='omit', alternative='two-sided')
        [tstat, vb_pValue] = scipy.stats.ttest_ind(np.ndarray.flatten(vbHoldWM), np.ndarray.flatten(vbHoldGM), equal_var=False, nan_policy='omit', alternative='two-sided')
        [tstat, taub_pValue] = scipy.stats.ttest_ind(np.ndarray.flatten(taubHoldWM), np.ndarray.flatten(taubHoldGM), equal_var=False, nan_policy='omit', alternative='two-sided')

        
        averagesArray = np.zeros([1,22])
        averagesArray[:,:2] = np.array([wmcount, gmcount])
        averagesArray[:,2:6] = wmgmmeans[0,:4]
        averagesArray[:,6:10] = wmgmstd[0,:4]
        averagesArray[:,10:14] = wmgmmeans[1,:4]
        averagesArray[:,14:18] = wmgmstd[1,:4]
        averagesArray[:,18:] = np.array([t1t_pValue,
                                     t1b_pValue, vb_pValue, taub_pValue])
        
''' ----------------------------SNR CRITERIA --------------------------------- ''' 

''' ----------------------------PLOTTING--------------------------------- ''' 


fig = plt.figure(constrained_layout= True)
fig.set_size_inches(30,15, forward='True')
fig.add_artist(lines.Line2D([0.2, 0.2], [0.1, 0.9],linewidth=5, linestyle='--', color='grey'))
subfigs = fig.subfigures(1, 2, wspace=0.05, hspace=0.05, width_ratios=[1, 4])

axs0 = subfigs[0].subplots(2, 1)
#subfigs[0].supxlabel('xlabel for subfigs[0]')

#T2
axs0[0].imshow(t2pad, cmap='gray',  interpolation='none')
axs0[0].set_title(r'$T_2-w$', x = 0.5, y=0.9)
axs0[0].get_xaxis().set_ticks([])
axs0[0].get_yaxis().set_ticks([])
axs0[0].spines['top'].set_visible(False)
axs0[0].spines['right'].set_visible(False)
axs0[0].spines['bottom'].set_visible(False)
axs0[0].spines['left'].set_visible(False)


#empty plot
axs0[1].get_xaxis().set_ticks([])
axs0[1].get_yaxis().set_ticks([])
axs0[1].spines['top'].set_visible(False)
axs0[1].spines['right'].set_visible(False)
axs0[1].spines['bottom'].set_visible(False)
axs0[1].spines['left'].set_visible(False) 

axs1 = subfigs[1].subplots(2, 3)
axs1[0,0].set_title('Extravascular T1')
#subfigs[1].supylabel('Quantitative Maps')

#T1t
if t1t == True:
    imaget1t = reg[0,:,:]
else: 
    imaget1t = t1tMaps_resized
axs1[0,0].imshow(t2image, cmap='gray',  interpolation='none') 
if lowBounds is True:
    im = axs1[0,0].imshow(imaget1t, cmap=colour_map1, alpha=1, interpolation='none', vmin= 400, vmax = 1600) 
else:
    im = axs1[0,0].imshow(imaget1t, cmap=colour_map1, alpha=1, interpolation='none', vmin= 600, vmax = 2100)  
cbar = plt.colorbar(im, ax=axs1[0,0], orientation='vertical', location='left', shrink = 0.5)
cbar.ax.set_title(r'$ms$')
axs1[0,0].set_title(r'Extravascular $T_1$')
axs1[0,0].get_xaxis().set_ticks([])
axs1[0,0].get_yaxis().set_ticks([])
axs1[0,0].spines['top'].set_visible(False)
axs1[0,0].spines['right'].set_visible(False)
axs1[0,0].spines['bottom'].set_visible(False)
axs1[0,0].spines['left'].set_visible(False)

#T1b
if t1b == True:
    imaget1b = reg[1,:,:]
else: 
    imaget1b = t1bMaps_resized
axs1[0,1].imshow(t2image, cmap='gray',  interpolation='none') 
if lowBounds is True:
    im = axs1[0,1].imshow(imaget1b, cmap=colour_map2, alpha=1, interpolation='none', vmin= 400, vmax = 2100) 
else:
    im = axs1[0,1].imshow(imaget1b, cmap=colour_map2, alpha=1, interpolation='none', vmin= 600, vmax = 2100)  
cbar = plt.colorbar(im, ax=axs1[0,1], orientation='vertical', location='left', shrink = 0.5)
axs1[0,1].set_title(r'Intravascular $T_1$')
cbar.ax.set_title(r'$ms$')
axs1[0,1].get_xaxis().set_ticks([])
axs1[0,1].get_yaxis().set_ticks([])
axs1[0,1].spines['top'].set_visible(False)
axs1[0,1].spines['right'].set_visible(False)
axs1[0,1].spines['bottom'].set_visible(False)
axs1[0,1].spines['left'].set_visible(False)

#B1+
if b1plus == True:
    imageb1 = reg[4,:,:]
else: 
    imageb1 = b1Maps_resized
axs1[0,2].imshow(t2image, cmap='gray',  interpolation='none') 
if lowBounds is True:
    im = axs1[0,2].imshow(imageb1, cmap=colour_map3, alpha=1, interpolation='none', vmin= 0.5, vmax = 1.0) 
else:
    im =axs1[0,2].imshow(imageb1, cmap=colour_map3, alpha=1, interpolation='none', vmin= 0.7, vmax = 1.2) 
cbar = plt.colorbar(im, ax=axs1[0,2], orientation='vertical', location='left', shrink = 0.5)
axs1[0,2].set_title(r'$B_1^+$')
axs1[0,2].get_xaxis().set_ticks([])
axs1[0,2].get_yaxis().set_ticks([])
axs1[0,2].spines['top'].set_visible(False)
axs1[0,2].spines['right'].set_visible(False)
axs1[0,2].spines['bottom'].set_visible(False)
axs1[0,2].spines['left'].set_visible(False)


#nu b
if vb == True:
    imagevb = np.abs(reg[2,:,:])
else: 
    imagevb = np.abs(vbMaps_resized)
axs1[1,0].imshow(t2image, cmap='gray',  interpolation='none') 
im = axs1[1,0].imshow(imagevb, cmap=colour_map4, alpha=1, interpolation='none', vmin= 1, vmax = 10) 
cbar = plt.colorbar(im, ax=axs1[1,0], orientation='vertical', location='left', shrink = 0.5)
cbar.ax.set_title(r'$\%$')
axs1[1,0].set_title(r'$\nu_b$')
axs1[1,0].get_xaxis().set_ticks([])
axs1[1,0].get_yaxis().set_ticks([])
axs1[1,0].spines['top'].set_visible(False)
axs1[1,0].spines['right'].set_visible(False)
axs1[1,0].spines['bottom'].set_visible(False)
axs1[1,0].spines['left'].set_visible(False)

#tau b
if taub == True:
    imagetaub = reg[3,:,:]
else: 
    imagetaub = taubMaps_resized
axs1[1,1].imshow(t2image, cmap='gray',  interpolation='none') 
im = axs1[1,1].imshow(imagetaub, cmap=colour_map5, alpha=1, interpolation='none', vmin= 200, vmax = 1600)
cbar = plt.colorbar(im, ax=axs1[1,1], orientation='vertical', location='left', shrink = 0.5)
cbar.ax.set_title(r'$ms$')
axs1[1,1].set_title(r'$\tau_b$')
axs1[1,1].get_xaxis().set_ticks([])
axs1[1,1].get_yaxis().set_ticks([])
axs1[1,1].spines['top'].set_visible(False)
axs1[1,1].spines['right'].set_visible(False)
axs1[1,1].spines['bottom'].set_visible(False)
axs1[1,1].spines['left'].set_visible(False)   

#RSS
if rssplus == True:
    imagerss = reg[5,:,:]
else: 
    imagerss = rssMaps_resized
axs1[1,2].imshow(t2image, cmap='gray',  interpolation='none') 
im = axs1[1,2].imshow(imagerss, cmap=colour_map6, alpha=1, interpolation='none', vmin= 0.0005, vmax = 0.01) 
axs1[1,2].set_title('RSS')
cbar = plt.colorbar(im, ax=axs1[1,2], orientation='vertical', location='left', shrink = 0.5)
axs1[1,2].get_xaxis().set_ticks([])
axs1[1,2].get_yaxis().set_ticks([])
axs1[1,2].spines['top'].set_visible(False)
axs1[1,2].spines['right'].set_visible(False)
axs1[1,2].spines['bottom'].set_visible(False)
axs1[1,2].spines['left'].set_visible(False) 
bbox = axs1[1,2].get_tightbbox(fig.canvas.get_renderer())
x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds
# slightly increase the very tight bounds:
xpad = 0.005 * width
ypad = 0.005 * height
fig.add_artist(plt.Rectangle((x0-xpad, y0-ypad), width+2*xpad, height+2*ypad, edgecolor='red', linewidth=3, fill=True))


if plotVert is True: 
    '''
    plt.style.use('bmh')
    plt.style.use('dark_background')
    #plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['figure.titlesize'] = 10
    '''
        
    fig, ax = plt.subplots(5)
    fig.set_size_inches(12,36, forward=True)
        
    plt.subplots_adjust(wspace=0, hspace=0)#(wspace=0, hspace=0)
    
    #T1t
    if t1t == True:
        imaget1t = reg[0,:,:]
    else: 
        imaget1t = t1tMaps_resized
    ax[0].imshow(t2image, cmap='gray',  interpolation='none') 
    im = ax[0].imshow(imaget1t, cmap=colour_map1, alpha=1, interpolation='none', vmin= 800, vmax = 2100) 
    cbar = plt.colorbar(im, ax=ax[0], orientation='vertical', location='left', shrink = 0.7)
    cbar.ax.set_title(r'$ms$')
    #ax[0].set_title(r'Extravascular $T_1$')
   # ax[0].set_ylabel(r'Extravascular $T_1$')
    ax[0].get_xaxis().set_ticks([])
    ax[0].get_yaxis().set_ticks([])
    ax[0]
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    
    #T1b
    if t1b == True:
        imaget1b = reg[1,:,:]
    else: 
        imaget1b = t1bMaps_resized
    ax[1].imshow(t2image, cmap='gray',  interpolation='none') 
    im = ax[1].imshow(imaget1b, cmap=colour_map2, alpha=1, interpolation='none', vmin= 800, vmax = 2100) 
    cbar = plt.colorbar(im, ax=ax[1], orientation='vertical', location='left', shrink = 0.7)
    #ax[1].set_title(r'Intravascular $T_1$')
    cbar.ax.set_title(r'$ms$')
    ax[1].get_xaxis().set_ticks([])
    ax[1].get_yaxis().set_ticks([])
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    
    #B1+
    ax[4].imshow(t2image, cmap='gray',  interpolation='none') 
    im = ax[4].imshow(b1Maps_resized, cmap=colour_map3, alpha=1, interpolation='none', vmin= 0.7, vmax = 1.2)
    cbar = plt.colorbar(im, ax=ax[4], orientation='vertical', location='left', shrink = 0.7)
    #ax[2].set_title(r'$B_1^+$')
    ax[4].get_xaxis().set_ticks([])
    ax[4].get_yaxis().set_ticks([])
    ax[4].spines['top'].set_visible(False)
    ax[4].spines['right'].set_visible(False)
    ax[4].spines['bottom'].set_visible(False)
    ax[4].spines['left'].set_visible(False)
    
    #nu b
    if vb == True:
        imagevb = reg[2,:,:]
    else: 
        imagevb = vbMaps_resized
    ax[3].imshow(t2image, cmap='gray',  interpolation='none') 
    im = ax[3].imshow(imagevb, cmap=colour_map4, alpha=1, interpolation='none', vmin= 1, vmax = 11) 
    cbar = plt.colorbar(im, ax=ax[3], orientation='vertical', location='left', shrink = 0.7)
    cbar.ax.set_title(r'$\%$')
    #ax[3].set_title(r'$\nu_b$')
    ax[3].get_xaxis().set_ticks([])
    ax[3].get_yaxis().set_ticks([])
    ax[3].spines['top'].set_visible(False)
    ax[3].spines['right'].set_visible(False)
    ax[3].spines['bottom'].set_visible(False)
    ax[3].spines['left'].set_visible(False)
    
    #tau b
    if taub == True:
        imagetaub = reg[3,:,:]
    else: 
        imagetaub = taubMaps_resized
    ax[2].imshow(t2image, cmap='gray',  interpolation='none') 
    im = ax[2].imshow(imagetaub, cmap=colour_map5, alpha=1, interpolation='none', vmin= 0, vmax = 1700) 
    cbar = plt.colorbar(im, ax=ax[2], orientation='vertical', location='left', shrink = 0.7)
    cbar.ax.set_title(r'$ms$')
    #ax[4].set_title(r'$\tau_b$')
    ax[2].get_xaxis().set_ticks([])
    ax[2].get_yaxis().set_ticks([])
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    ax[2].spines['bottom'].set_visible(False)
    ax[2].spines['left'].set_visible(False)

'''--------------------------------CALCULATE AVERAGES----------------------------'''

#grey matter masks

gmMaskHold = 2*atlas_resize_s/2
#Remove WM from mask

gmMaskHold[gmMaskHold==2] = 0 
gmMaskHold[gmMaskHold==41] = 0 
gmMaskHold[gmMaskHold<1000] = 0 

gmMaskHold[gmMaskHold>0] = 1 
gmMaskHold = gmMaskHold*(1-binary_seg_mask_CSF)

gmcount = np.sum(gmMaskHold)

fig, ax = plt.subplots()
fig.set_size_inches(20,15, forward=True)
plt.imshow((gmMaskHold), cmap='Greys_r')
cbar = plt.colorbar()
cbar.set_label('Segmented GM')

grey_t1t_mask = gmMaskHold*t1tMaps_resized*snrmaskplot
grey_t1b_mask = gmMaskHold*t1bMaps_resized*snrmaskplot
grey_taub_mask = gmMaskHold*taubMaps_resized*snrmaskplot
grey_vb_mask = gmMaskHold*vbMaps_resized*snrmaskplot
#grey_b1_mask = binary_seg_gm*b1Maps

#nan all zero values
grey_t1t_mask[grey_t1t_mask==0] = np.nan
grey_t1b_mask[grey_t1b_mask==0] = np.nan
grey_taub_mask[grey_taub_mask==0] = np.nan
grey_vb_mask[grey_vb_mask ==0] = np.nan

#grey matter averages and std
grey_t1t_mean = np.nanmean(grey_t1t_mask[np.nonzero(grey_t1t_mask)])
grey_t1t_std = np.nanstd(grey_t1t_mask[np.nonzero(grey_t1t_mask)])
grey_t1t_mask = grey_t1t_mask[~np.isnan(grey_t1t_mask)]
grey_t1t_iqr = scipy.stats.iqr(grey_t1t_mask[np.nonzero(grey_t1t_mask)])

grey_t1b_mean = np.nanmean(grey_t1b_mask[np.nonzero(grey_t1b_mask)])
grey_t1b_std = np.nanstd(grey_t1b_mask[np.nonzero(grey_t1b_mask)])
grey_t1b_mask = grey_t1b_mask[~np.isnan(grey_t1b_mask)]
grey_t1b_iqr = scipy.stats.iqr(grey_t1b_mask[np.nonzero(grey_t1b_mask)])

grey_taub_mean = np.nanmean(grey_taub_mask[np.nonzero(grey_taub_mask)])
grey_taub_std = np.nanstd(grey_taub_mask[np.nonzero(grey_taub_mask)])
grey_taub_mask = grey_taub_mask[~np.isnan(grey_taub_mask)]
grey_taub_iqr = scipy.stats.iqr(grey_taub_mask[np.nonzero(grey_taub_mask)])

grey_vb_mean = np.nanmean(grey_vb_mask[np.nonzero(grey_vb_mask)])
grey_vb_std = np.nanstd(grey_vb_mask[np.nonzero(grey_vb_mask)])
grey_vb_mask = grey_vb_mask[~np.isnan(grey_vb_mask)]
grey_vb_iqr = scipy.stats.iqr(grey_vb_mask[np.nonzero(grey_vb_mask)])
'''
grey_b1_mean = grey_b1_mask[np.nonzero(grey_b1_mask)].mean()
grey_b1_std = grey_b1_mask[np.nonzero(grey_b1_mask)].std()
'''

wmMaskHold = np.zeros([res_x, res_y])
#Remove WM from mask
wmMaskHold[atlas_resize_s==2] = 1
wmMaskHold[atlas_resize_s==41] = 1
wmMaskHold = wmMaskHold*(1-binary_seg_mask_CSF)

wmcount = np.sum(wmMaskHold)

fig, ax = plt.subplots()
fig.set_size_inches(20,15, forward=True)
plt.imshow((wmMaskHold), cmap='Greys_r')
cbar = plt.colorbar()
cbar.set_label('Segmented WM')

#white matter masks
white_t1t_mask = wmMaskHold*t1tMaps_resized*snrmaskplot
white_t1b_mask = wmMaskHold*t1bMaps_resized*snrmaskplot
white_taub_mask = wmMaskHold*taubMaps_resized*snrmaskplot
white_vb_mask = wmMaskHold*vbMaps_resized*snrmaskplot
#white_b1_mask = wmMaskHold*b1Maps

#nan all zero values
white_t1t_mask[white_t1t_mask==0] = np.nan
white_t1b_mask[white_t1b_mask==0] = np.nan
white_taub_mask[white_taub_mask==0] = np.nan
white_vb_mask[white_vb_mask ==0] = np.nan

#white matter averages and std
white_t1t_mean = np.nanmean(white_t1t_mask[np.nonzero(white_t1t_mask)])
white_t1t_std = np.nanstd(white_t1t_mask[np.nonzero(white_t1t_mask)])
white_t1t_mask = white_t1t_mask[~np.isnan(white_t1t_mask)]
white_t1t_iqr = scipy.stats.iqr(white_t1t_mask[np.nonzero(white_t1t_mask)])

white_t1b_mean = np.nanmean(white_t1b_mask[np.nonzero(white_t1b_mask)])
white_t1b_std = np.nanstd(white_t1b_mask[np.nonzero(white_t1b_mask)])
white_t1b_mask = white_t1b_mask[~np.isnan(white_t1b_mask)]
white_t1b_iqr = scipy.stats.iqr(white_t1b_mask[np.nonzero(white_t1b_mask)])

white_taub_mean = np.nanmean(white_taub_mask[np.nonzero(white_taub_mask)])
white_taub_std = np.nanstd(white_taub_mask[np.nonzero(white_taub_mask)])
white_taub_mask = white_taub_mask[~np.isnan(white_taub_mask)]
white_taub_iqr = scipy.stats.iqr(white_taub_mask[np.nonzero(white_taub_mask)])

white_vb_mean = np.nanmean(white_vb_mask[np.nonzero(white_vb_mask)])
white_vb_std = np.nanstd(white_vb_mask[np.nonzero(white_vb_mask)])
white_vb_mask = white_vb_mask[~np.isnan(white_vb_mask)]
white_vb_iqr = scipy.stats.iqr(white_vb_mask[np.nonzero(white_vb_mask)])

'''
white_b1_mean = white_b1_mask[np.nonzero(white_b1_mask)].mean()
white_b1_std = white_b1_mask[np.nonzero(white_b1_mask)].std()
'''
'''-------------------------------- T-TEST----------------------------''' 

[tstat, t1t_pValue] = scipy.stats.ttest_ind(np.ndarray.flatten(white_t1t_mask), np.ndarray.flatten(grey_t1t_mask), equal_var=False, nan_policy='omit', alternative='two-sided')
[tstat, t1b_pValue] = scipy.stats.ttest_ind(np.ndarray.flatten(white_t1b_mask), np.ndarray.flatten(grey_t1b_mask), equal_var=False, nan_policy='omit', alternative='two-sided')
[tstat, vb_pValue] = scipy.stats.ttest_ind(np.ndarray.flatten(white_vb_mask), np.ndarray.flatten(grey_vb_mask), equal_var=False, nan_policy='omit', alternative='two-sided')
[tstat, taub_pValue] = scipy.stats.ttest_ind(np.ndarray.flatten(white_taub_mask), np.ndarray.flatten(grey_taub_mask), equal_var=False, nan_policy='omit', alternative='two-sided')


'''-------------------------------- PRINT MEANS----------------------------'''

print('Mean T1t: WM - ' + str(white_t1t_mean) + ' +- ' +  str(white_t1t_std))
print('Mean T1t: GM - ' + str(grey_t1t_mean) + ' +- ' +  str(grey_t1t_std)) 
print('p =' +  str(np.round(t1t_pValue,3)))
print(' ')
print('Mean T1b: WM - ' + str(white_t1b_mean) + ' +- ' +  str(white_t1b_std))
print('Mean T1b: GM - ' + str(grey_t1b_mean) + ' +- ' +  str(grey_t1b_std))
print('p =' +  str(np.round(t1b_pValue,3)))
print(' ')
print('Mean vb: WM - ' + str(white_vb_mean) + ' +- ' +  str(white_vb_std))
print('Mean vb: GM - ' + str(grey_vb_mean) + ' +- ' +  str(grey_vb_std))
print('p =' +  str(np.round(vb_pValue,3)))
print(' ')
print('Mean taub: WM - ' + str(white_taub_mean) + ' +- ' +  str(white_taub_std))
print('Mean taub: GM - ' + str(grey_taub_mean) + ' +- ' +  str(grey_taub_std))
print('p =' +  str(np.round(taub_pValue,3)))
print(' ')

averagesArray = np.zeros([1,22])
averagesArray[:] = np.array([wmcount, gmcount,
                 white_t1t_mean,white_t1b_mean,white_vb_mean,white_taub_mean,
                 white_t1t_std,white_t1b_std,white_vb_std,white_taub_std,
                 grey_t1t_mean,grey_t1b_mean,grey_vb_mean,grey_taub_mean,
                 grey_t1t_std,grey_t1b_std,grey_vb_std,grey_taub_std,
                 t1t_pValue,t1b_pValue, vb_pValue, taub_pValue]).T

iqrArray = np.zeros([1,8])
iqrArray[:] = np.array([white_t1t_iqr,white_t1b_iqr,white_vb_iqr,white_taub_iqr,
                      grey_t1t_iqr,grey_t1b_iqr,grey_vb_iqr,grey_taub_iqr])
'''-------------------------------- SAVE AVERAGES----------------------------'''

if ica is True:
    filename = 'RegionalAveragesICA.csv'
elif hydra is True: 
    filename = 'RegionalAveragesHYDRA.csv'
else: 
    filename = 'RegionalAverages.csv'
  
np.savetxt(os.path.join(pathToFolder, filename), matches, delimiter =", ", fmt ='% s')

#Stop timer and print                                                    
t1 = time.time()
total = t1-t0
print("Total Time:    " +  str(total) + " seconds")  


'''
pathtofold = ('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + 
    str(volunteer_no) +'/Anatomy_Seg/')
pathtoT1 = (pathtofold + 'T1_seg.nii.gz')

t1image = nib.load(pathtoT1)._data
t1image = t1image[:,:,26]
new_image = nib.Nifti1Image(t1image, affine=nib.load(pathtoT1).affine)

nib.save(new_image, os.path.join(pathtofold, 'T1_slice.nii.gz'))  
'''

