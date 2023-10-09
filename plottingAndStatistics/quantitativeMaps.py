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
import skimage.io
import skimage.color
import skimage.filters
import nibabel as nib
from skimage.transform import resize
import cv2
import scipy
import warnings
warnings.filterwarnings("ignore")

#Set plotting style 
plt.style.use('dark_background')
plt.rcParams.update({'font.size': 50})
plt.rcParams['font.family'] = 'Times'

import cmcrameri.cm as cmc
#from cmcrameri import show_cmaps
#show_cmaps()
colourHold =  cmc.devon
colour_map1 = colour_map2 = colour_map3 = colour_map4 = colour_map5 = colourHold
colour_map6 =  cmc.lajolla_r 

import time 

t0 = time.time()

#go up a folder
os.chdir("..")
print(os.getcwd())

''' -----------------------------INPUTS--------------------------------- '''

#Input volunteer number
volunteer_no = 1.1

# Dictionary folder
dictfolder = 'WEXandBVfreeNew'  #'SliceProfileNew' #'WEXandBVfreeNew' #'Sequence' #'InversionRecovery'

#image resolutiopn  
res_x =  64; res_y = 64;
#length of acquisiton
acqlen = 2000

#Number of paramters in the atlas labels - overestimation is okay, its just for array size 
par = 3000
#Slice of the atlas that corresponds to the slice on the MRF images
sli = int(138/2)

#Folder Path
# Image folder paths 
pathToFolder = ('./SampleData/Volunteer' + str(volunteer_no) + '/MRF')
bigpath = (pathToFolder + '/Maps/') 


# Dictionary folder
dictfolder = 'SliceProfileNew' #'WEXandBVfreeNew' #'SliceProfileNew'

dictPath = ('./Dictionaries/Dictionary' + dictfolder + '/')

#is the 'b1-first' matching technique used? 
b1first = True


if b1first is True: 
    bigpath = (bigpath + 'B1first_')

#is denoised data used?
denoise = False
filt = 'G'
#for gaussian filter
sig = 0.5
    
#Is the data being shown voxel-wise or regionally
#if regionally atlas load is True
atlasLoad = True

#Which paramters are regional 
if atlasLoad is True: 
    t1t = t1b = vb = taub = True
    b1plus = rssplus = False
else: 
    t1t = t1b = vb = taub = b1plus = rsspluse = False



# SNR fit failure 
#Exclude voxels with SNR below the threshold from calcualtions
snrFF = False 
#threshold 
snrFFcriteria = 3.2

#Display masked images? 
mask_im = 'yes'
#Segement images?
seg_im = 'yes'

'''--------------------------------READ IN MAPS----------------------------'''

print("Starting Read in:  " + str(time.strftime('%X %x %Z')))
   
#Load in maps
if denoise is True: 
    if filt == 'G':
        t1tMaps = np.rot90(nib.load((bigpath + 'T1_t[ms]_' + dictfolder + '_G_S_' + str(sig) +'.nii.gz'))._dataobj)
        t1bMaps = np.rot90(nib.load((bigpath + 'T1_b[ms]_' + dictfolder + '_G_S_' + str(sig) +'.nii.gz'))._dataobj)
        taubMaps = np.rot90(nib.load((bigpath + 'tau_b[ms]_' + dictfolder + '_G_S_' + str(sig) +'.nii.gz'))._dataobj)
        vbMaps = np.rot90(nib.load((bigpath + 'v_b[%]_' + dictfolder + '_G_S_' + str(sig) +'.nii.gz'))._dataobj)
        b1Maps = np.rot90(nib.load((bigpath + 'B1+_' + dictfolder + '_G_S_' + str(sig) +'.nii.gz'))._dataobj)
        snr = np.rot90(nib.load((bigpath + 'SNR_' + dictfolder + '_G_S_' + str(sig) +'.nii.gz'))._dataobj)
        rssMaps = np.rot90(nib.load((bigpath + 'Residual_' + dictfolder + '_G_S_' + str(sig) +'.nii.gz'))._dataobj)
    elif filt == 'MF':
        t1tMaps = np.rot90(nib.load((bigpath + 'T1_t[ms]_' + dictfolder + '_MF_S_3_.nii.gz'))._dataobj)
        t1bMaps = np.rot90(nib.load((bigpath + 'T1_b[ms]_' + dictfolder + '_MF_S_3_.nii.gz'))._dataobj)
        taubMaps = np.rot90(nib.load((bigpath + 'tau_b[ms]_' + dictfolder + '_MF_S_3_.nii.gz'))._dataobj)
        vbMaps = np.rot90(nib.load((bigpath + 'v_b[%]_' + dictfolder + '_MF_S_3_.nii.gz'))._dataobj)
        b1Maps = np.rot90(nib.load((bigpath + 'B1+_' + dictfolder + '_MF_S_3_.nii.gz'))._dataobj)
        snr = np.rot90(nib.load((bigpath + 'SNR_' + dictfolder + '_MF_S_3_.nii.gz'))._dataobj)
        rssMaps = np.rot90(nib.load((bigpath + 'Residual_' + dictfolder + '_MF_S_3_.nii.gz'))._dataobj)
else:
    t1tMaps = np.rot90(nib.load((bigpath + 'T1_t[ms]_' + dictfolder + '.nii.gz'))._dataobj)
    t1bMaps = np.rot90(nib.load((bigpath + 'T1_b[ms]_' + dictfolder + '.nii.gz'))._dataobj)
    taubMaps = np.rot90(nib.load((bigpath + 'tau_b[ms]_' + dictfolder + '.nii.gz'))._dataobj)
    vbMaps = np.rot90(nib.load((bigpath + 'v_b[%]_' + dictfolder + '.nii.gz'))._dataobj)
    b1Maps = np.rot90(nib.load((bigpath + 'B1+_' + dictfolder + '.nii.gz'))._dataobj)
    snr = np.rot90(nib.load((bigpath + 'SNR_' + dictfolder + '.nii.gz'))._dataobj)
    rssMaps = np.rot90(nib.load((bigpath + 'Residual_' + dictfolder + '.nii.gz'))._dataobj)
 
'''--------------------------------LOAD T2 IMAGE----------------------------'''

pathtoT2 = ('./SampleData/Volunteer' + str(volunteer_no) +'/Mask_Images/T2_reg.nii.gz')

t2bigimage = (np.rot90(np.squeeze((nib.load(pathtoT2)._dataobj))))

t2image = cv2.resize(t2bigimage, dsize=(res_x, res_y), interpolation=cv2.INTER_NEAREST)

#pad T2 image for plotting  
t2pad =  np.pad(t2bigimage, pad_width=((40, 0), (10, 0)))

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
    for filename in glob.glob(os.path.join(str('./SampleData/Volunteer' + str(volunteer_no) + '/Mask_Images/' + rr))):
        
        mask_load = nib.load(filename)
        mask = (np.flipud(np.array(mask_load.dataobj).T))
        
        mask_resized = cv2.resize(mask, dsize=(res_x, res_y), interpolation=cv2.INTER_NEAREST)
        
        try:
            grey_image = skimage.color.rgb2grey(mask_resized)
        except: 
            grey_image = mask_resized
        histogram, bin_edges = np.histogram(grey_image, bins=256, range=(0.0, 1.0))
        t = 3e-05
        binary_mask = mask_resized > t
        binary_mask_csf = binary_mask.astype('uint8')

''' -----------------------SEGMENTATION-------------------------- '''

if seg_im == 'yes':  
    seg_path = './SampleData/Volunteer' + str(volunteer_no) + '/Mask_Images/Segmented/'
    
    segCSF = 0
    
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
    snrMask = np.zeros(np.shape(snr))
    snrMask[snr>snrFFcriteria] = 1 
    
    fig, ax = plt.subplots()
    plt.imshow(t2image,cmap='gray')
    
    snrmaskplot = snrMask*binary_mask
    snrmask = (1-snrMask)*binary_mask_csf
    snrmask[snrmask==0] = np.NaN
    
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
    
    subAtlasFile = ('./SampleData/Volunteer' + str(volunteer_no) +'/Anatomy_Seg/outputAtlas.nii.gz')
    
    subAtlasLoad = nib.load(subAtlasFile)
    subAtlas = np.array(subAtlasLoad.dataobj)
    
    atlasSliceSub = np.round((np.flipud(subAtlas[:,:,sli].T)),0)

    atlas = atlasSliceSub
    
    atlas_resize_s = cv2.resize(atlas, dsize=(res_x, res_y), interpolation=cv2.INTER_NEAREST)
    atlas_resize_l = cv2.resize(atlas, dsize=(np.shape(t2bigimage)), interpolation=cv2.INTER_NEAREST)
          

    '''--------------------------------PLOT OVERLAY----------------------------'''
    fig, ax = plt.subplots()
    plt.imshow(t2bigimage,cmap='gray')
    atlasRound = np.round(atlas_resize_l,0)
    pltatlas = (atlasRound)*mask
    
    #pltatlas[np.isnan(pltatlas)] = 0
    plt.imshow(np.log(pltatlas), alpha=1, cmap = colour_map1)


if atlasLoad is True:
    ''' ---------------------EXTRACT REGIONAL AVERAGES--------------------------- '''  
    regSigs = np.zeros([par, 6, res_x ,res_x])
    matches = [['Index', 'T1t' , 'T1b', 'vb', 'taub','b1', 'RSS']]
        
    for i in range(1,par): 

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
        
''' ----------------------------PLOTTING--------------------------------- ''' 


fig = plt.figure(constrained_layout= True)
fig.set_size_inches(30,15, forward='True')
fig.add_artist(lines.Line2D([0.2, 0.2], [0.1, 0.9],linewidth=5, linestyle='--', color='grey'))
subfigs = fig.subfigures(1, 2, wspace=0.05, hspace=0.05, width_ratios=[1, 4])

axs0 = subfigs[0].subplots(2, 1)

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
im =axs1[0,2].imshow(imageb1, cmap=colour_map3, alpha=1, interpolation='none', vmin= 0.7, vmax = 1.2) 
cbar = plt.colorbar(im, ax=axs1[0,2], orientation='vertical', location='left', shrink = 0.5)
axs1[0,2].set_title(r'$B_1^+$')
axs1[0,2].get_xaxis().set_ticks([])
axs1[0,2].get_yaxis().set_ticks([])
axs1[0,2].spines['top'].set_visible(False)
axs1[0,2].spines['right'].set_visible(False)
axs1[0,2].spines['bottom'].set_visible(False)
axs1[0,2].spines['left'].set_visible(False)


#vb
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

#Stop timer and print                                                    
t1 = time.time()
total = t1-t0
print("Total Time:    " +  str(total) + " seconds")  

