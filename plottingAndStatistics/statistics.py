#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""------------------------------------------------------------------------

Statistical analysis code containing:
- Repeatability Coefficient (RC) AND intra-class correlation coefficent (ICC) between scan and rescan  
- Bland-Altmann plots between scan-rescan for each parameter (T1t, T1b, vb, and tb)
- Mean bar chart for WM and GM across each parameter
- T-test between WM/GM for each parameter

Author: Emma Thomson
Year: 2023
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
import scipy
import pingouin as pg
import pandas as pd

plt.rcParams.update({'font.size': 24})
plt.rcParams['font.family'] = 'Times'

import warnings
warnings.filterwarnings("ignore")

#go up a folder
os.chdir("..")

''' -----------------------------FUNCTIONS--------------------------------- '''

#Flatten list of lists into a single list 
def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])

#Bland-Altmann plot code 
def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    CI_low    = md - 1.96*sd
    CI_high   = md + 1.96*sd
    
    return md, sd, mean, diff, CI_low, CI_high

#Adds the significance bars and asterisks to the bar chart
def barplot_annotate_brackets(num1, num2, data, center, height, axnum, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p and p>0.001:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1] + height[num1]/5
    rx, ry = center[num2], height[num2] + height[num2]/5

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    ax[axnum].plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    ax[axnum].text(*mid, text, **kwargs)


''' -----------------------------INPUTS--------------------------------- '''

#The volunteers you wish to include in the statistical analysis
#For this code to work each volunteer needs scan and rescan images
volunteerRange = [1] #list(range(1,11)) #inclusive

#Length of the acquisition
acqlen = 2000

#Do you use denoised data? 
denoise = False 
#if you are denoising then what type of filter do you want to use 
filt = 'G' #G = gaussian #MF = median filtering
#if gaussian filter - how much smoothing do you want
sigma = 0.5 #1 

#Do you used B1-first matching?
b1first = True

#dictionary folder name 
dictfolder = 'SliceProfileNew'

#Display masked images? 
mask_im = 'yes'
#Save parameter maps? 
save_im = 'yes'
#Segement images?
seg_im = 'no'

#Is the data being shown voxel-wise or regionally
#if regionally atlas load is True
atlasLoad = False

#Which paramters are regional 
if atlasLoad is True: 
    t1t = t1b = vb = taub = True
    b1plus = rssplus = False
else: 
    t1t = t1b = vb = taub = b1plus = rsspluse = False

#image resolution 
res_x =  64; res_y = 64;


''' -----------------------------LOAD MAPS--------------------------------- '''

#Empty array to occupy for each parameter: number of parameters, 4??, scan and rescan 
WMmeans = np.zeros([np.size(volunteerRange,0),4,2])
GMmeans = np.zeros([np.size(volunteerRange,0),4,2])
WMstd = np.zeros([np.size(volunteerRange,0),4,2])
GMstd = np.zeros([np.size(volunteerRange,0),4,2])

#Loop over volunteers
for volit in range(np.size(volunteerRange,0)):  
    
    #Loop over repeats 
    for rep in range(2):
        
        #gives the 0.1 and 0.2 for scan and repeat
        repit = (rep+1)/10
         
        volunteer_no = volunteerRange[volit] + repit
        
        '''--------------------------------READ IN MAPS----------------------------'''

        #Folder Path
        #Image folder paths 
        pathToFolder = ('./sampleData/Volunteer' + str(volunteer_no) + '/MRF')
        if b1first is True:
            bigpath = (pathToFolder + '/Maps/B1first_') 
        else: 
            bigpath = (pathToFolder + '/Maps') 
    
        t1tMaps = np.rot90(nib.load((bigpath + 'T1_t[ms]_' + dictfolder + '.nii.gz'))._dataobj)
        t1bMaps = np.rot90(nib.load((bigpath + 'T1_b[ms]_' + dictfolder + '.nii.gz'))._dataobj)
        taubMaps = np.rot90(nib.load((bigpath + 'tau_b[ms]_' + dictfolder + '.nii.gz'))._dataobj)
        vbMaps = np.rot90(nib.load((bigpath + 'v_b[%]_' + dictfolder + '.nii.gz'))._dataobj)
        b1Maps = np.rot90(nib.load((bigpath + 'B1+_' + dictfolder + '.nii.gz'))._dataobj)
        if denoise is True: 
            if filt == 'G':
                t1tMaps = np.rot90(nib.load((bigpath + 'T1_t[ms]_' + dictfolder + '_G_S_' + str(sigma) + '.nii.gz'))._dataobj)
                t1bMaps = np.rot90(nib.load((bigpath + 'T1_b[ms]_' + dictfolder + '_G_S_' + str(sigma) + '.nii.gz'))._dataobj)
                taubMaps = np.rot90(nib.load((bigpath + 'tau_b[ms]_' + dictfolder + '_G_S_' + str(sigma) + '.nii.gz'))._dataobj)
                vbMaps = np.rot90(nib.load((bigpath + 'v_b[%]_' + dictfolder + '_G_S_' + str(sigma) + '.nii.gz'))._dataobj)
                b1Maps = np.rot90(nib.load((bigpath + 'B1+_' + dictfolder + '_G_S_' + str(sigma) + '.nii.gz'))._dataobj)
            elif filt == 'MF':
                t1tMaps = np.rot90(nib.load((bigpath + 'T1_t[ms]_' + dictfolder + '_MF_S_3.nii.gz'))._dataobj)
                t1bMaps = np.rot90(nib.load((bigpath + 'T1_b[ms]_' + dictfolder + '_MF_S_3.nii.gz'))._dataobj)
                taubMaps = np.rot90(nib.load((bigpath + 'tau_b[ms]_' + dictfolder + '_MF_S_3.nii.gz'))._dataobj)
                vbMaps = np.rot90(nib.load((bigpath + 'v_b[%]_' + dictfolder + '_MF_S_3.nii.gz'))._dataobj)
                b1Maps = np.rot90(nib.load((bigpath + 'B1+_' + dictfolder + '_MF_S_3.nii.gz'))._dataobj)
     

        '''--------------------------------RESIZE QMAPS----------------------------'''
        
        x,y = res_x, res_y
        
        t1tMaps_resized = resize(t1tMaps, (x, y), anti_aliasing=True)
        t1bMaps_resized = resize(t1bMaps, (x, y), anti_aliasing=True)
        taubMaps_resized = resize(taubMaps, (x, y), anti_aliasing=True)
        vbMaps_resized = resize(vbMaps, (x, y), anti_aliasing=True)
        b1Maps_resized = resize(b1Maps, (x, y), anti_aliasing=True)
        
        ''' -----------------------MASK BRAIN-------------------------- '''

        rr = '*brain_mask.nii.gz' 
        for filename in glob.glob(os.path.join(str('./sampleData/Volunteer' + str(volunteer_no) + '/Mask_Images/' + rr))):
            
            mask_load = nib.load(filename)
            mask = np.flipud(np.array(mask_load.dataobj).T)
            
            mask_resized = cv2.resize(mask, dsize=(res_x, res_y), interpolation=cv2.INTER_NEAREST)
            #mask = mask.resize([res_x, res_y])
        
            gray_image =  mask_resized #skimage.color.rgb2gray(mask_resized)
            #blurred_image = skimage.filters.gaussian(gray_image, sigma=1.0)
            histogram, bin_edges = np.histogram(gray_image, bins=256, range=(0.0, 1.0))
            t = 3e-05
            binary_mask = mask_resized > t
            binary_mask = binary_mask.astype('uint8') #int(binary_mask == 'True')
            #binary_mask = abs(np.int64(binary_mask)-1)
    
          
        seg_path = './sampleData/Volunteer' + str(volunteer_no) + '/Mask_Images/Segmented/'
        
        segCSF = 0
        
        #Remove any CSF component
        segCSF = (seg_path + 'T2_pve_' + str(segCSF) + '.nii.gz')
        fpath =  segCSF
        seg_load_CSF = nib.load(fpath) 
        segCSF = np.array(seg_load_CSF.dataobj)
        if volunteer_no != 17.1:
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
        
        binary_mask = binary_mask-binary_seg_mask_CSF*binary_mask
        
        t1tMaps_resized = t1tMaps_resized*binary_mask
        t1bMaps_resized = t1bMaps_resized*binary_mask
        taubMaps_resized = taubMaps_resized*binary_mask
        vbMaps_resized = vbMaps_resized*binary_mask
        b1Maps_resized = b1Maps_resized*binary_mask
    
        #nan all zero values
        t1tMaps_resized[t1tMaps_resized==0] = np.nan
        t1bMaps_resized[t1bMaps_resized==0] = np.nan
        taubMaps_resized[taubMaps_resized==0] = np.nan
        vbMaps_resized[vbMaps_resized ==0] = np.nan
        b1Maps_resized[b1Maps_resized==0] = np.nan

        ''' --------------------------READ IN ATLAS------------------------------ '''  
    
        subAtlasFile = ('./sampleData/Volunteer' + str(volunteer_no) +'/Anatomy_Seg/outputAtlas.nii.gz')
        
        subAtlasLoad = nib.load(subAtlasFile)
        subAtlas = np.array(subAtlasLoad.dataobj)
        
        atlasSliceSub = np.round((np.flipud(subAtlas[:,:,int(138/2)].T)),0)
        
        atlas = atlasSliceSub
        
        atlas_resize_s = cv2.resize(atlas, dsize=(res_x, res_y), interpolation=cv2.INTER_NEAREST)    
    
        gmMaskHold = 2*atlas_resize_s/2
    
        #Remove WM from mask
        gmMaskHold[gmMaskHold==2] = 0 
        gmMaskHold[gmMaskHold==41] = 0 
        gmMaskHold[gmMaskHold<1000] = 0 
        
        gmMaskHold[gmMaskHold>0] = 1 
        gmMaskHold = gmMaskHold*(1-binary_seg_mask_CSF)
    
        wmMaskHold = np.zeros([res_x, res_y])
        #Remove WM from mask
        wmMaskHold[atlas_resize_s==2] = 1
        wmMaskHold[atlas_resize_s==41] = 1
        wmMaskHold = wmMaskHold*(1-binary_seg_mask_CSF)
    
        
        ''' -----------------------------PLOTTING--------------------------------- '''
        
        grey_t1t_mask = gmMaskHold*t1tMaps_resized
        grey_t1b_mask = gmMaskHold*t1bMaps_resized
        grey_taub_mask = gmMaskHold*taubMaps_resized
        grey_vb_mask = gmMaskHold*vbMaps_resized
        
        #nan all zero values
        grey_t1t_mask[grey_t1t_mask==0] = np.nan
        grey_t1b_mask[grey_t1b_mask==0] = np.nan
        grey_taub_mask[grey_taub_mask==0] = np.nan
        grey_vb_mask[grey_vb_mask ==0] = np.nan
    
        #grey matter averages and std
        grey_t1t_mean = np.nanmedian(grey_t1t_mask[np.nonzero(grey_t1t_mask)])
        grey_t1t_std = np.std(grey_t1t_mask[np.nonzero(grey_t1t_mask)])
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
    
        #white matter masks
        white_t1t_mask = wmMaskHold*t1tMaps_resized
        white_t1b_mask = wmMaskHold*t1bMaps_resized
        white_taub_mask = wmMaskHold*taubMaps_resized
        white_vb_mask = wmMaskHold*vbMaps_resized
    
        #nan all zero values
        white_t1t_mask[white_t1t_mask==0] = np.nan
        white_t1b_mask[white_t1b_mask==0] = np.nan
        white_taub_mask[white_taub_mask==0] = np.nan
        white_vb_mask[white_vb_mask==0] = np.nan
        
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
        
        WMmeans[volit,:,rep] = np.array([white_t1t_mean, white_t1b_mean, white_vb_mean, white_taub_mean])      
        GMmeans[volit,:,rep] = np.array([grey_t1t_mean, grey_t1b_mean, grey_vb_mean, grey_taub_mean]) 
        WMstd[volit,:,rep] = np.array([white_t1t_std, white_t1b_std, white_vb_std, white_taub_std])  
        GMstd[volit,:,rep] = np.array([grey_t1t_std, grey_t1b_std, grey_vb_std, grey_taub_std])

'''------------------------ REPEATIBILITY COEFFICIENT-------------------------''' 
 
rcWM = 1.96 * np.sqrt(np.sum((WMmeans[:,:,0]-WMmeans[:,:,1])**2, axis=0)/np.size(volunteerRange))
rcGM = 1.96 * np.sqrt(np.sum((GMmeans[:,:,0]-GMmeans[:,:,1])**2, axis=0)/np.size(volunteerRange))

'''----------------INTRACLASS CORRELATION COEFFICIENT------------------------''' 

#Reformat data as dataframe 

#num params, wm and gm
ICC = np.zeros([4,2])

for param in range(4):
    WMmeansParam = pd.DataFrame({'volunteer': np.float64(list(np.array(range(0,np.size((volunteerRange))))) + list(np.array(range(0,np.size((volunteerRange)))))),
                                 'group': (['S']*np.size(volunteerRange) + ['RS']*np.size(volunteerRange)),
                            'Param': (flatten(list(WMmeans[:,param,0]) + list(WMmeans[:,param,1])))})
                              
    
    iccWMParam = pg.intraclass_corr(data=WMmeansParam, targets='volunteer', raters='group', ratings='Param')
    
    GMmeansParam = pd.DataFrame({'volunteer': np.float64(list(np.array(range(0,np.size((volunteerRange))))) + list(np.array(range(0,np.size((volunteerRange)))))),
                                 'group': (['S']*np.size(volunteerRange) + ['RS']*np.size(volunteerRange)),
                            'Param': (flatten(list(GMmeans[:,param,0]) + list(GMmeans[:,param,1])))})
                              
    
    iccGMParam = pg.intraclass_corr(data=GMmeansParam, targets='volunteer', raters='group', ratings='Param')

    ICC[param,0] = np.array(iccWMParam['ICC'])[0]
    ICC[param,1] = np.array(iccGMParam['ICC'])[0]

''' -----------------------------BLAND ALTMANN--------------------------------- '''

## WHITE MATTER BLAND ALTMANN

fig, ax = plt.subplots(2,2)
fig.set_size_inches(10,8, forward=True)

plt.subplots_adjust(wspace=0.3, hspace=0.3)

colours = ['red', 'blue', 'green', 'black', 'orange','cyan','magenta', 'brown', 'pink', 'purple', 'orange']

for iter in range(np.size(volunteerRange,0)):
    
    #T1t
    md, sd, mean, diff, CI_low, CI_high = bland_altman_plot(WMmeans[:,0,0].flatten('F'), WMmeans[:,0,1].flatten('F'))
    for i in range(np.size(volunteerRange,0)):
        ax[0,0].scatter(mean[i], diff[i], marker='x', c=colours[i])
        
        #ax[0,0].scatter(mean[numberOfRepeats+(i)], diff[numberOfRepeats+(i)], marker='x', c=colours[i])

    ax[0,0].axhline(md,           color='black', linestyle='-')
    ax[0,0].axhline(md + 1.96*sd, color='gray', linestyle='--')
    ax[0,0].axhline(md - 1.96*sd, color='gray', linestyle='--')
    ax[0,0].set_xlabel(r'Mean $T_{1,t}$ [ms]')
    ax[0,0].set_ylabel('Difference [ms]')

    #T1b
    md, sd, mean, diff, CI_low, CI_high = bland_altman_plot(WMmeans[:,1,0].flatten('F'), WMmeans[:,1,1].flatten('F'))
    for i in range(np.size(volunteerRange,0)):
        ax[0,1].scatter(mean[i], diff[i], marker='x', c=colours[i])
        
        #ax[0,1].scatter(mean[numberOfRepeats+(i)], diff[numberOfRepeats+(i)], marker='x', c=colours[i])

    ax[0,1].axhline(md,           color='black', linestyle='-')
    ax[0,1].axhline(md + 1.96*sd, color='gray', linestyle='--')
    ax[0,1].axhline(md - 1.96*sd, color='gray', linestyle='--')
    ax[0,1].set_xlabel(r'Mean $T_{1,b}$ [ms]')
    ax[0,1].set_ylabel('Difference [ms]')
    
    #vb
    md, sd, mean, diff, CI_low, CI_high = bland_altman_plot(WMmeans[:,2,0].flatten('F'), WMmeans[:,2,1].flatten('F'))
    for i in range(np.size(volunteerRange,0)):
        ax[1,0].scatter(mean[i], diff[i], marker='x', c=colours[i])
        
        #ax[1,0].scatter(mean[numberOfRepeats+(i)], diff[numberOfRepeats+(i)], marker='x', c=colours[i])

    ax[1,0].axhline(md,           color='black', linestyle='-')
    ax[1,0].axhline(md + 1.96*sd, color='gray', linestyle='--')
    ax[1,0].axhline(md - 1.96*sd, color='gray', linestyle='--')
    ax[1,0].set_xlabel(r'Mean $\nu_{b}$ [ms]')
    ax[1,0].set_ylabel('Difference [%]')
    
    #taub
    md, sd, mean, diff, CI_low, CI_high = bland_altman_plot(WMmeans[:,3,0].flatten('F'), WMmeans[:,3,1].flatten('F'))
    for i in range(np.size(volunteerRange,0)):
        ax[1,1].scatter(mean[i], diff[i], marker='x', c=colours[i])
        
        #ax[1,1].scatter(mean[numberOfRepeats+(i)], diff[numberOfRepeats+(i)], marker='x', c=colours[i])

    ax[1,1].axhline(md,           color='black', linestyle='-')
    ax[1,1].axhline(md + 1.96*sd, color='gray', linestyle='--')
    ax[1,1].axhline(md - 1.96*sd, color='gray', linestyle='--')
    ax[1,1].set_xlabel(r'Mean $\tau_{b}$ [ms]')
    ax[1,1].set_ylabel('Difference [ms]')
    
## GREY MATTER BLAND ALTMANN

fig, ax = plt.subplots(2,2)
fig.set_size_inches(10,8, forward=True)

plt.subplots_adjust(wspace=0.3, hspace=0.3)

colours = ['red', 'blue', 'green', 'black', 'orange','cyan','magenta', 'brown', 'pink', 'purple', 'orange']

for iter in range(np.size(volunteerRange,0)):
    
    #T1t
    md, sd, mean, diff, CI_low, CI_high = bland_altman_plot(GMmeans[:,0,0].flatten('F'), GMmeans[:,0,1].flatten('F'))
    for i in range(np.size(volunteerRange,0)):
        ax[0,0].scatter(mean[i], diff[i], marker='x', c=colours[i])
        
        #ax[0,0].scatter(mean[numberOfRepeats+(i)], diff[numberOfRepeats+(i)], marker='x', c=colours[i])

    ax[0,0].axhline(md,           color='black', linestyle='-')
    ax[0,0].axhline(md + 1.96*sd, color='gray', linestyle='--')
    ax[0,0].axhline(md - 1.96*sd, color='gray', linestyle='--')
    ax[0,0].set_xlabel(r'Mean $T_{1,t}$ [ms]')
    ax[0,0].set_ylabel('Difference [ms]')

    #T1b
    md, sd, mean, diff, CI_low, CI_high = bland_altman_plot(GMmeans[:,1,0].flatten('F'), GMmeans[:,1,1].flatten('F'))
    for i in range(np.size(volunteerRange,0)):
        ax[0,1].scatter(mean[i], diff[i], marker='x', c=colours[i])
        
        #ax[0,1].scatter(mean[numberOfRepeats+(i)], diff[numberOfRepeats+(i)], marker='x', c=colours[i])

    ax[0,1].axhline(md,           color='black', linestyle='-')
    ax[0,1].axhline(md + 1.96*sd, color='gray', linestyle='--')
    ax[0,1].axhline(md - 1.96*sd, color='gray', linestyle='--')
    ax[0,1].set_xlabel(r'Mean $T_{1,b}$ [ms]')
    ax[0,1].set_ylabel('Difference [ms]')
    
    #vb
    md, sd, mean, diff, CI_low, CI_high = bland_altman_plot(GMmeans[:,2,0].flatten('F'), GMmeans[:,2,1].flatten('F'))
    for i in range(np.size(volunteerRange,0)):
        ax[1,0].scatter(mean[i], diff[i], marker='x', c=colours[i])
        
        #ax[1,0].scatter(mean[numberOfRepeats+(i)], diff[numberOfRepeats+(i)], marker='x', c=colours[i])

    ax[1,0].axhline(md,           color='black', linestyle='-')
    ax[1,0].axhline(md + 1.96*sd, color='gray', linestyle='--')
    ax[1,0].axhline(md - 1.96*sd, color='gray', linestyle='--')
    ax[1,0].set_xlabel(r'Mean $\nu_{b}$ [ms]')
    ax[1,0].set_ylabel('Difference [%]')
    
    #taub
    md, sd, mean, diff, CI_low, CI_high = bland_altman_plot(GMmeans[:,3,0].flatten('F'), GMmeans[:,3,1].flatten('F'))
    for i in range(np.size(volunteerRange,0)):
        ax[1,1].scatter(mean[i], diff[i], marker='x', c=colours[i])
        
        #ax[1,1].scatter(mean[numberOfRepeats+(i)], diff[numberOfRepeats+(i)], marker='x', c=colours[i])

    ax[1,1].axhline(md,           color='black', linestyle='-')
    ax[1,1].axhline(md + 1.96*sd, color='gray', linestyle='--')
    ax[1,1].axhline(md - 1.96*sd, color='gray', linestyle='--')
    ax[1,1].set_xlabel(r'Mean $\tau_{b}$ [ms]')
    ax[1,1].set_ylabel('Difference [ms]')
    
    
'''-------------------------------- T-TEST----------------------------''' 
#Concatenate all volunteers and repeats into one array
WMconcat = np.concatenate((WMmeans[:,:,0], WMmeans[:,:,1]), axis=0)
GMconcat = np.concatenate((GMmeans[:,:,0], GMmeans[:,:,1]), axis=0)


[tstat, t1t_pValue] = scipy.stats.ttest_ind(WMconcat[:,0], GMconcat[:,0], equal_var=False, nan_policy='omit', alternative='two-sided')
[tstat, t1b_pValue] = scipy.stats.ttest_ind(WMconcat[:,1], GMconcat[:,1], equal_var=False, nan_policy='omit', alternative='two-sided')
[tstat, vb_pValue] = scipy.stats.ttest_ind(WMconcat[:,2], GMconcat[:,2], equal_var=False, nan_policy='omit', alternative='two-sided')
[tstat, taub_pValue] = scipy.stats.ttest_ind(WMconcat[:,3], GMconcat[:,3], equal_var=False, nan_policy='omit', alternative='two-sided')

WMstd =  np.std(WMconcat, axis=0)
WMmeans = np.mean(WMconcat, axis=0)

GMstd =  np.std(GMconcat, axis=0)
GMmeans = np.mean(GMconcat, axis=0)

''' -----------------------------PLOT BAR CHART --------------------------------- '''


fig,ax = plt.subplots(1,4)
fig.set_size_inches(11,5, forward='True')
plt.subplots_adjust(wspace=1, hspace=10)#(wspace=0, hspace=0)
#ax = fig.add_axes([0, 0, 1, 1])

xpos=  np.arange(2)
labels = ['WM', 'GM']

ax[0].bar(0, WMmeans[0], yerr=WMstd[0], capsize=10, color='darkgrey', width=1)
ax[0].bar(1, GMmeans[0], yerr=GMstd[0], capsize=10, color='dimgrey', width=1)
ax[0].get_xaxis().set_ticks(xpos)
ax[0].set_xticklabels(labels)
ax[0].set_ylim([0, 2000])
ax[0].set_ylabel(r'$T_{1,t}$ [ms]')
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
if t1t_pValue <= 0.05:
    barplot_annotate_brackets(0, 1, t1t_pValue, xpos, [WMmeans[0],GMmeans[0]], 0)


ax[1].bar(0, WMmeans[1], yerr=WMstd[1], capsize=10, color='darkgrey', width=1)
ax[1].bar(1, GMmeans[1], yerr=GMstd[1], capsize=10, color='dimgrey', width=1)
ax[1].get_xaxis().set_ticks(xpos)
ax[1].set_xticklabels(labels)
ax[1].set_ylim([0,2000])
ax[1].set_ylabel(r'$T_{1,b}$ [ms]')
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
if t1b_pValue <= 0.05:
    barplot_annotate_brackets(0, 1, t1b_pValue, xpos, [WMmeans[1],GMmeans[1]], 1)


ax[2].bar(0, WMmeans[2], yerr=WMstd[2], capsize=10, color='darkgrey', width=1)
ax[2].bar(1, GMmeans[2], yerr=GMstd[2], capsize=10, color='dimgrey', width=1)
ax[2].get_xaxis().set_ticks(xpos)
ax[2].set_xticklabels(labels)
ax[2].set_ylabel(r'$\nu_{b}$ [%]')
ax[2].set_ylim([0,4])
ax[2].spines['top'].set_visible(False)
ax[2].spines['right'].set_visible(False)
if vb_pValue <= 0.05:
    barplot_annotate_brackets(0, 1, vb_pValue, xpos, [WMmeans[2],GMmeans[2]+0.2], 2)

ax[3].bar(0, WMmeans[3], yerr=WMstd[3], capsize=10, color='darkgrey', width=1)
ax[3].bar(1, GMmeans[3], yerr=GMstd[3], capsize=10, color='dimgrey', width=1)
ax[3].get_xaxis().set_ticks(xpos)
ax[3].set_xticklabels(labels)
ax[3].set_ylim([0,1200])
ax[3].set_ylabel(r'$\tau_{b}$  [ms]')
ax[3].spines['top'].set_visible(False)
ax[3].spines['right'].set_visible(False)
if taub_pValue <= 0.05:
    barplot_annotate_brackets(0, 1, taub_pValue, xpos, [WMmeans[3],GMmeans[3]], 3)
    
    
## FIX ME: ADD SOME PRINT STATEMENTS