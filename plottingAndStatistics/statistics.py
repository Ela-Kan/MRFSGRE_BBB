 #!/usr/bin/env python3
 # -*- coding: utf-8 -*-
 """------------------------------------------------------------------------

BIG STATS CODE
- RC AND ICC 
- BLAND ALTMANN
- MEAN BAR CHART 
- T TEST

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
 import csv
 import scipy
 import pingouin as pg
 import pandas as pd


 #plt.style.use('bmh')
 plt.rcParams.update({'font.size': 24})
 plt.rcParams['font.family'] = 'Times'

 ''' -----------------------------FUNCTIONS--------------------------------- '''

 def flatten(list_of_lists):
     if len(list_of_lists) == 0:
         return list_of_lists
     if isinstance(list_of_lists[0], list):
         return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
     return list_of_lists[:1] + flatten(list_of_lists[1:])

def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    CI_low    = md - 1.96*sd
    CI_high   = md + 1.96*sd
    
    '''
    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='black', linestyle='-')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    '''
    return md, sd, mean, diff, CI_low, CI_high

''' -----------------------------FUNCTIONS--------------------------------- '''

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

 volunteerRange = [22,23,24,25,26,27,28,29,30,31] #inclusive
 #corresponding to the order of the volunteers 
 #numberOfRepeats = 2 #number of scans per HV. 1 scan + 1 repeat = 2 

 #Other imaging factors 
 Regime = 'Red' #Ext
 acqlen = 2000
 number_of_readouts = 4
 TE = '2_LARGE_IRFAT'
 readout = "VDS" #'VDS' 

 parNo = 2000

 hydra = False
 denoise = False 
 b1first = True
 ica = False

 flip_maps1 = False 
 flip_maps2 = False

 #average or rematch 
 rematch = False

 plotVert = True

 #if you want just subcortical atlas then 1, if both cortical and subcortical then 2
 atlasNo = 1

 #Display masked images? 
 mask_im = 'no'
 #Save parameter maps? 
 save_im = 'yes'
 #Segement images?
 seg_im = 'no'
 #bias correction? - DONT THINK THIS IS GOOD
 bias_corr = 'no'

     
 #compare what? #left vs right #white vs gray matter #all components
 subCort = False #Subcortical matching only 

 ## TO DO: Add additional identifier here for  name of folder

 #number of regions 
 par = 82

 #sometimes the maps are the wrong orientation. idk why
 flip_maps = False

 #what is being parcellized 
 t1t = False
 t1b = False 
 vb = False
 taub = False
 b1plus = False

 #image resolution 
 res_x =  64; res_y = 64;


 ''' -----------------------------LOAD MAPS--------------------------------- '''

 #Number of FA is less than acquisition length because we want to get rid of initial values 
 #initial_bin = 1
 #no_of_FA = acqlen-initial_bin
 #data_offset = 1
 #Number of entries in dictionary (to set array sizes)
 #components = [6,6,6,10,7,6,6,5,11,7] #[6,6,6,10,7,6,6,5,11,7] 

 WMmeans = np.zeros([np.size(volunteerRange,0),4,2])
 GMmeans = np.zeros([np.size(volunteerRange,0),4,2])
 WMstd = np.zeros([np.size(volunteerRange,0),4,2])
 GMstd = np.zeros([np.size(volunteerRange,0),4,2])

 rep = 1
 
 #Loop over volunteers
 for volit in range(np.size(volunteerRange,0)):     

     '''--------------------------------READ IN MAPS----------------------------'''
     
     #First scan
     volunteer_no = volunteerRange[volit] + 0.1
     #Folder Path
     # Image folder paths 
     pathToFolder = ('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + 
         str(volunteer_no) + '/' + str(Regime) + str(acqlen) + '_' + str(readout) +
         str(number_of_readouts) + '_TE' + str(TE))
     if ica is True:
         bigpath = (pathToFolder + '/Maps_ICA/') 
     elif hydra is True:
         bigpath = (pathToFolder + '/Maps_HYDRA/') 
     elif b1first is True:
         bigpath = (pathToFolder + '/Maps/B1first_') 
     
     t1tMaps1 = np.rot90(nib.load((bigpath + 'T1_t[ms]_SliceProfileNew.nii.gz'))._dataobj)
     t1bMaps1 = np.rot90(nib.load((bigpath + 'T1_b[ms]_SliceProfileNew.nii.gz'))._dataobj)
     taubMaps1 = np.rot90(nib.load((bigpath + 'tau_b[ms]_SliceProfileNew.nii.gz'))._dataobj)
     vbMaps1 = np.rot90(nib.load((bigpath + 'v_b[%]_SliceProfileNew.nii.gz'))._dataobj)
     b1Maps1 = np.rot90(nib.load((bigpath + 'B1+_SliceProfileNew.nii.gz'))._dataobj)
     if denoise is True: 
         t1tMaps1 = np.rot90(nib.load((bigpath + 'T1_t[ms]_SliceProfileNew_G_S_0.5.nii.gz'))._dataobj)
         t1bMaps1 = np.rot90(nib.load((bigpath + 'T1_b[ms]_SliceProfileNew_G_S_0.5.nii.gz'))._dataobj)
         taubMaps1 = np.rot90(nib.load((bigpath + 'tau_b[ms]_SliceProfileNew_G_S_0.5.nii.gz'))._dataobj)
         vbMaps1 = np.rot90(nib.load((bigpath + 'v_b[%]_SliceProfileNew_G_S_0.5.nii.gz'))._dataobj)
         b1Maps1 = np.rot90(nib.load((bigpath + 'B1+_SliceProfileNew_G_S_0.5.nii.gz'))._dataobj)
      
     #Second scan
     volunteer_no = volunteerRange[volit] + 0.2
     pathToFolder = ('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + 
            str(volunteer_no) + '/' + str(Regime) + str(acqlen) + '_' + str(readout) +
            str(number_of_readouts) + '_TE' + str(TE))
     if ica is True:
            bigpath = (pathToFolder + '/Maps_ICA/') 
     elif hydra is True:
            bigpath = (pathToFolder + '/Maps_HYDRA/') 
     elif b1first is True:
            bigpath = (pathToFolder + '/Maps/B1first_') 
     
     t1tMaps2 = np.rot90(nib.load((bigpath + 'T1_t[ms]_SliceProfileNew.nii.gz'))._dataobj)
     t1bMaps2 = np.rot90(nib.load((bigpath + 'T1_b[ms]_SliceProfileNew.nii.gz'))._dataobj)
     taubMaps2 = np.rot90(nib.load((bigpath + 'tau_b[ms]_SliceProfileNew.nii.gz'))._dataobj)
     vbMaps2 = np.rot90(nib.load((bigpath + 'v_b[%]_SliceProfileNew.nii.gz'))._dataobj)
     b1Maps2 = np.rot90(nib.load((bigpath + 'B1+_SliceProfileNew.nii.gz'))._dataobj)
        
     if ica is True: 
         components = [6,6,6,10,7,6,6,5,11,7] 
         t1tMaps2 = np.rot90(nib.load((bigpath + 'T1_t[ms]_components' + str(components[volit]) + 'SliceProfileNew.nii.gz'))._dataobj)
         t1bMaps2 = np.rot90(nib.load((bigpath + 'T1_b[ms]_components' + str(components[volit]) + 'SliceProfileNew.nii.gz'))._dataobj)
         taubMaps2 = np.rot90(nib.load((bigpath + 'tau_b[ms]_components' + str(components[volit]) + 'SliceProfileNew.nii.gz'))._dataobj)
         vbMaps2 = np.rot90(nib.load((bigpath + 'v_b[%]_components' + str(components[volit]) + 'SliceProfileNew.nii.gz'))._dataobj)
         b1Maps2 = np.rot90(nib.load((bigpath + 'B1+_components' + str(components[volit]) + 'SliceProfileNew.nii.gz'))._dataobj)
     if denoise is True:
         t1tMaps2 = np.rot90(nib.load((bigpath + 'T1_t[ms]_SliceProfileNew_G_S_0.5.nii.gz'))._dataobj)
         t1bMaps2 = np.rot90(nib.load((bigpath + 'T1_b[ms]_SliceProfileNew_G_S_0.5.nii.gz'))._dataobj)
         taubMaps2 = np.rot90(nib.load((bigpath + 'tau_b[ms]_SliceProfileNew_G_S_0.5.nii.gz'))._dataobj)
         vbMaps2 = np.rot90(nib.load((bigpath + 'v_b[%]_SliceProfileNew_G_S_0.5.nii.gz'))._dataobj)
         b1Maps2 = np.rot90(nib.load((bigpath + 'B1+_SliceProfileNew_G_S_0.5.nii.gz'))._dataobj)
         
     
     if flip_maps1 is True: 
         t1tMaps1 = np.fliplr(t1tMaps1)
         t1bMaps1 = np.fliplr(t1bMaps1)
         taubMaps1 = np.fliplr(taubMaps1)
         vbMaps1 = np.fliplr(vbMaps1)
         b1Maps1 = np.fliplr(b1Maps1)
     if flip_maps2 is True:
         t1tMaps2 = np.fliplr(t1tMaps2)
         t1bMaps2 = np.fliplr(t1bMaps2)
         taubMaps2 = np.fliplr(taubMaps2)
         vbMaps2 = np.fliplr(vbMaps2)
         b1Maps2 = np.fliplr(b1Maps2)
      

     '''--------------------------------RESIZE QMAPS----------------------------'''
     
     x,y = res_x, res_y
     
     t1tMaps_resized1 = resize(t1tMaps1, (x, y), anti_aliasing=True)
     t1bMaps_resized1 = resize(t1bMaps1, (x, y), anti_aliasing=True)
     taubMaps_resized1 = resize(taubMaps1, (x, y), anti_aliasing=True)
     vbMaps_resized1 = resize(vbMaps1, (x, y), anti_aliasing=True)
     b1Maps_resized1 = resize(b1Maps1, (x, y), anti_aliasing=True)
     t1tMaps_resized2 = resize(t1tMaps2, (x, y), anti_aliasing=True)
     t1bMaps_resized2= resize(t1bMaps2, (x, y), anti_aliasing=True)
     taubMaps_resized2 = resize(taubMaps2, (x, y), anti_aliasing=True)
     vbMaps_resized2 = resize(vbMaps2, (x, y), anti_aliasing=True)
     b1Maps_resized2 = resize(b1Maps2, (x, y), anti_aliasing=True)
     #m0Maps_resized = resize(m0Maps, (x, y), anti_aliasing=True)
     
     ''' -----------------------MASK BRAIN-------------------------- '''
     
     #First scan
     volunteer_no = volunteerRange[volit] + 0.1
     rr = '*brain_mask.nii.gz' 
     for filename in glob.glob(os.path.join(str('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + str(volunteer_no) + '/Mask_Images/' + rr))):
         
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
     binary_seg_mask_CSF1 = binary_seg_mask.astype('uint8')
     
     binary_mask1 = binary_mask-binary_seg_mask_CSF1*binary_mask
     
     #Second scan
     volunteer_no = volunteerRange[volit] + 0.2
     rr = '*brain_mask.nii.gz' 
     for filename in glob.glob(os.path.join(str('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + str(volunteer_no) + '/Mask_Images/' + rr))):
         
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
     binary_seg_mask_CSF2 = binary_seg_mask.astype('uint8')
     
     binary_mask2 = binary_mask-binary_seg_mask_CSF2*binary_mask
     
     t1tMaps_resized1 = t1tMaps_resized1*binary_mask1
     t1bMaps_resized1 = t1bMaps_resized1*binary_mask1
     taubMaps_resized1 = taubMaps_resized1*binary_mask1
     vbMaps_resized1 = vbMaps_resized1*binary_mask1
     b1Maps_resized1 = b1Maps_resized1*binary_mask1
     t1tMaps_resized2 = t1tMaps_resized2*binary_mask2
     t1bMaps_resized2 = t1bMaps_resized2*binary_mask2
     taubMaps_resized2 = taubMaps_resized2*binary_mask2
     vbMaps_resized2 = vbMaps_resized2*binary_mask2
     b1Maps_resized2 = b1Maps_resized2*binary_mask2
     #m0Maps_resized = m0Maps_resized*binary_mask
     
     #nan all zero values
     t1tMaps_resized1[t1tMaps_resized1==0] = np.nan
     t1bMaps_resized1[t1bMaps_resized1==0] = np.nan
     taubMaps_resized1[taubMaps_resized1==0] = np.nan
     vbMaps_resized1[vbMaps_resized1 ==0] = np.nan
     b1Maps_resized1[b1Maps_resized1==0] = np.nan
     t1tMaps_resized2[t1tMaps_resized2==0] = np.nan
     t1bMaps_resized2[t1bMaps_resized2==0] = np.nan
     taubMaps_resized2[taubMaps_resized2==0] = np.nan
     vbMaps_resized2[vbMaps_resized2 ==0] = np.nan
     b1Maps_resized2[b1Maps_resized2==0] = np.nan
     #m0Maps_resized[m0Maps_resized==0] = np.nan

     '''
     fig,ax = plt.subplots()
     plt.imshow(t1tMaps_resized1)
     fig,ax = plt.subplots()
     plt.imshow(t1tMaps_resized2)
     '''
     ''' --------------------------READ IN ATLAS------------------------------ '''  

     #First scan 
     volunteer_no = volunteerRange[volit] + 0.1
     #subAtlasFile = ('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + 
     #    str(volunteer_no) +'/Anatomy_Seg/outputAtlas.nii.gz')
     subAtlasFile = ('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + 
         str(volunteer_no) +'/Anatomy_Seg/outputAtlas.nii.gz')
     
     subAtlasLoad = nib.load(subAtlasFile)
     subAtlas = np.array(subAtlasLoad.dataobj)
     
     atlasSliceSub = np.round((np.flipud(subAtlas[:,:,int(138/2)].T)),0)
     
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
         atlasSliceCort = np.round((np.flipud(cortAtlas[:,:,int(138/2)].T)),0)
         #atlasSliceCort = scipy.signal.medfilt(atlasSliceCort, kernel_size=5)
         #To remove overlap with atlasSliceSub segs
         atlasSliceCort += 20
         
         atlas = atlasSliceSub + atlasSliceCort
         
     else: 
         atlas = atlasSliceSub
     
     #atlas = scipy.signal.medfilt(atlas, kernel_size=9)
     atlas_resize_s1 = cv2.resize(atlas, dsize=(res_x, res_y), interpolation=cv2.INTER_NEAREST)
     
     #Second scan 
     volunteer_no = volunteerRange[volit] + 0.2
     #subAtlasFile = ('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + 
     #    str(volunteer_no) +'/Anatomy_Seg/outputAtlas.nii.gz')
     subAtlasFile = ('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + 
         str(volunteer_no) +'/Anatomy_Seg/outputAtlas.nii.gz')
     
     subAtlasLoad = nib.load(subAtlasFile)
     subAtlas = np.array(subAtlasLoad.dataobj)
     
     atlasSliceSub = np.round((np.flipud(subAtlas[:,:,int(138/2)].T)),0)
     
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
         atlasSliceCort = np.round((np.flipud(cortAtlas[:,:,int(138/2)].T)),0)
         #atlasSliceCort = scipy.signal.medfilt(atlasSliceCort, kernel_size=5)
         #To remove overlap with atlasSliceSub segs
         atlasSliceCort += 20
         
         atlas = atlasSliceSub + atlasSliceCort
         
     else: 
         atlas = atlasSliceSub
     
     #atlas = scipy.signal.medfilt(atlas, kernel_size=9)
     atlas_resize_s2 = cv2.resize(atlas, dsize=(res_x, res_y), interpolation=cv2.INTER_NEAREST)

     gmMaskHold1 = 2*atlas_resize_s1/2
     gmMaskHold2 = 2*atlas_resize_s2/2
     
     #Remove WM from mask
     gmMaskHold1[gmMaskHold1==2] = 0 
     gmMaskHold1[gmMaskHold1==41] = 0 
     gmMaskHold1[gmMaskHold1<1000] = 0 
     
     gmMaskHold2[gmMaskHold2==2] = 0 
     gmMaskHold2[gmMaskHold2==41] = 0 
     gmMaskHold2[gmMaskHold2<1000] = 0 
     
     gmMaskHold1[gmMaskHold1>0] = 1 
     gmMaskHold1 = gmMaskHold1*(1-binary_seg_mask_CSF1)

     gmMaskHold2[gmMaskHold2>0] = 1 
     gmMaskHold2 = gmMaskHold2*(1-binary_seg_mask_CSF2)
     
     
     wmMaskHold1 = np.zeros([res_x, res_y])
     #Remove WM from mask
     wmMaskHold1[atlas_resize_s1==2] = 1
     wmMaskHold1[atlas_resize_s1==41] = 1
     wmMaskHold1 = wmMaskHold1*(1-binary_seg_mask_CSF1)
     
     wmMaskHold2 = np.zeros([res_x, res_y])
     #Remove WM from mask
     wmMaskHold2[atlas_resize_s2==2] = 1
     wmMaskHold2[atlas_resize_s2==41] = 1
     wmMaskHold2 = wmMaskHold2*(1-binary_seg_mask_CSF2)
     
     ''' -----------------------------PLOTTING--------------------------------- '''
     
     grey_t1t_mask1 = gmMaskHold1*t1tMaps_resized1
     grey_t1b_mask1 = gmMaskHold1*t1bMaps_resized1
     grey_taub_mask1 = gmMaskHold1*taubMaps_resized1
     grey_vb_mask1 = gmMaskHold1*vbMaps_resized1
     
     grey_t1t_mask2 = gmMaskHold2*t1tMaps_resized2
     grey_t1b_mask2 = gmMaskHold2*t1bMaps_resized2
     grey_taub_mask2 = gmMaskHold2*taubMaps_resized2
     grey_vb_mask2 = gmMaskHold2*vbMaps_resized2
     #grey_b1_mask = binary_seg_gm*b1Maps
     
     #nan all zero values
     grey_t1t_mask1[grey_t1t_mask1==0] = np.nan
     grey_t1b_mask1[grey_t1b_mask1==0] = np.nan
     grey_taub_mask1[grey_taub_mask1==0] = np.nan
     grey_vb_mask1[grey_vb_mask1 ==0] = np.nan
     
     grey_t1t_mask2[grey_t1t_mask2==0] = np.nan
     grey_t1b_mask2[grey_t1b_mask2==0] = np.nan
     grey_taub_mask2[grey_taub_mask2==0] = np.nan
     grey_vb_mask2[grey_vb_mask2 ==0] = np.nan
     
     #grey matter averages and std
     grey_t1t_mean1 = np.nanmedian(grey_t1t_mask1[np.nonzero(grey_t1t_mask1)])
     grey_t1t_std1 = np.std(grey_t1t_mask1[np.nonzero(grey_t1t_mask1)])
     grey_t1t_mask1 = grey_t1t_mask1[~np.isnan(grey_t1t_mask1)]
     grey_t1t_iqr1 = scipy.stats.iqr(grey_t1t_mask1[np.nonzero(grey_t1t_mask1)])
     
     grey_t1t_mean2 = np.nanmean(grey_t1t_mask2[np.nonzero(grey_t1t_mask2)])
     grey_t1t_std2 = np.nanstd(grey_t1t_mask2[np.nonzero(grey_t1t_mask2)])
     grey_t1t_mask2 = grey_t1t_mask2[~np.isnan(grey_t1t_mask2)]
     grey_t1t_iqr2 = scipy.stats.iqr(grey_t1t_mask2[np.nonzero(grey_t1t_mask2)])
     
     grey_t1b_mean1 = np.nanmean(grey_t1b_mask1[np.nonzero(grey_t1b_mask1)])
     grey_t1b_std1 = np.nanstd(grey_t1b_mask1[np.nonzero(grey_t1b_mask1)])
     grey_t1b_mask1 = grey_t1b_mask1[~np.isnan(grey_t1b_mask1)]
     grey_t1b_iqr1 = scipy.stats.iqr(grey_t1b_mask1[np.nonzero(grey_t1b_mask1)])
     
     grey_t1b_mean2 = np.nanmean(grey_t1b_mask2[np.nonzero(grey_t1b_mask2)])
     grey_t1b_std2 = np.nanstd(grey_t1b_mask2[np.nonzero(grey_t1b_mask2)])
     grey_t1b_mask2 = grey_t1b_mask2[~np.isnan(grey_t1b_mask2)]
     grey_t1b_iqr2 = scipy.stats.iqr(grey_t1b_mask2[np.nonzero(grey_t1b_mask2)])
     
     grey_taub_mean1 = np.nanmean(grey_taub_mask1[np.nonzero(grey_taub_mask1)])
     grey_taub_std1 = np.nanstd(grey_taub_mask1[np.nonzero(grey_taub_mask1)])
     grey_taub_mask1 = grey_taub_mask1[~np.isnan(grey_taub_mask1)]
     grey_taub_iqr1 = scipy.stats.iqr(grey_taub_mask1[np.nonzero(grey_taub_mask1)])
     
     grey_taub_mean2 = np.nanmean(grey_taub_mask2[np.nonzero(grey_taub_mask2)])
     grey_taub_std2 = np.nanstd(grey_taub_mask2[np.nonzero(grey_taub_mask2)])
     grey_taub_mask2 = grey_taub_mask2[~np.isnan(grey_taub_mask2)]
     grey_taub_iqr2 = scipy.stats.iqr(grey_taub_mask2[np.nonzero(grey_taub_mask2)])
     
     grey_vb_mean1 = np.nanmean(grey_vb_mask1[np.nonzero(grey_vb_mask1)])
     grey_vb_std1 = np.nanstd(grey_vb_mask1[np.nonzero(grey_vb_mask1)])
     grey_vb_mask1 = grey_vb_mask1[~np.isnan(grey_vb_mask1)]
     grey_vb_iqr1 = scipy.stats.iqr(grey_vb_mask1[np.nonzero(grey_vb_mask1)])
     
     grey_vb_mean2 = np.nanmean(grey_vb_mask2[np.nonzero(grey_vb_mask2)])
     grey_vb_std2 = np.nanstd(grey_vb_mask2[np.nonzero(grey_vb_mask2)])
     grey_vb_mask2 = grey_vb_mask2[~np.isnan(grey_vb_mask2)]
     grey_vb_iqr2 = scipy.stats.iqr(grey_vb_mask2[np.nonzero(grey_vb_mask2)])


     #white matter masks
     white_t1t_mask1 = wmMaskHold1*t1tMaps_resized1
     white_t1b_mask1 = wmMaskHold1*t1bMaps_resized1
     white_taub_mask1 = wmMaskHold1*taubMaps_resized1
     white_vb_mask1 = wmMaskHold1*vbMaps_resized1  
     
     white_t1t_mask2 = wmMaskHold2*t1tMaps_resized2
     white_t1b_mask2 = wmMaskHold2*t1bMaps_resized2
     white_taub_mask2 = wmMaskHold2*taubMaps_resized2
     white_vb_mask2 = wmMaskHold2*vbMaps_resized2
     #white_b1_mask = wmMaskHold*b1Maps

     #nan all zero values
     white_t1t_mask1[white_t1t_mask1==0] = np.nan
     white_t1b_mask1[white_t1b_mask1==0] = np.nan
     white_taub_mask1[white_taub_mask1==0] = np.nan
     white_vb_mask1[white_vb_mask1==0] = np.nan
     
     white_t1t_mask2[white_t1t_mask2==0] = np.nan
     white_t1b_mask2[white_t1b_mask2==0] = np.nan
     white_taub_mask2[white_taub_mask2==0] = np.nan
     white_vb_mask2[white_vb_mask2==0] = np.nan
     
     #white matter averages and std
     white_t1t_mean1 = np.nanmean(white_t1t_mask1[np.nonzero(white_t1t_mask1)])
     white_t1t_std1 = np.nanstd(white_t1t_mask1[np.nonzero(white_t1t_mask1)])
     white_t1t_mask1 = white_t1t_mask1[~np.isnan(white_t1t_mask1)]
     white_t1t_iqr1 = scipy.stats.iqr(white_t1t_mask1[np.nonzero(white_t1t_mask1)])
     
     white_t1t_mean2 = np.nanmean(white_t1t_mask2[np.nonzero(white_t1t_mask2)])
     white_t1t_std2 = np.nanstd(white_t1t_mask2[np.nonzero(white_t1t_mask2)])
     white_t1t_mask2 = white_t1t_mask2[~np.isnan(white_t1t_mask2)]
     white_t1t_iqr2 = scipy.stats.iqr(white_t1t_mask2[np.nonzero(white_t1t_mask2)])
     
     white_t1b_mean1 = np.nanmean(white_t1b_mask1[np.nonzero(white_t1b_mask1)])
     white_t1b_std1 = np.nanstd(white_t1b_mask1[np.nonzero(white_t1b_mask1)])
     white_t1b_mask1 = white_t1b_mask1[~np.isnan(white_t1b_mask1)]
     white_t1b_iqr1 = scipy.stats.iqr(white_t1b_mask1[np.nonzero(white_t1b_mask1)])
     
     white_t1b_mean2 = np.nanmean(white_t1b_mask2[np.nonzero(white_t1b_mask2)])
     white_t1b_std2 = np.nanstd(white_t1b_mask2[np.nonzero(white_t1b_mask2)])
     white_t1b_mask2 = white_t1b_mask2[~np.isnan(white_t1b_mask2)]
     white_t1b_iqr2 = scipy.stats.iqr(white_t1b_mask2[np.nonzero(white_t1b_mask2)])
     
     white_taub_mean1 = np.nanmean(white_taub_mask1[np.nonzero(white_taub_mask1)])
     white_taub_std1 = np.nanstd(white_taub_mask1[np.nonzero(white_taub_mask1)])
     white_taub_mask1 = white_taub_mask1[~np.isnan(white_taub_mask1)]
     white_taub_iqr1 = scipy.stats.iqr(white_taub_mask1[np.nonzero(white_taub_mask1)])
     
     white_taub_mean2 = np.nanmean(white_taub_mask2[np.nonzero(white_taub_mask2)])
     white_taub_std2 = np.nanstd(white_taub_mask2[np.nonzero(white_taub_mask2)])
     white_taub_mask2 = white_taub_mask2[~np.isnan(white_taub_mask2)]
     white_taub_iqr2 = scipy.stats.iqr(white_taub_mask2[np.nonzero(white_taub_mask2)])
     
     white_vb_mean1 = np.nanmean(white_vb_mask1[np.nonzero(white_vb_mask1)])
     white_vb_std1 = np.nanstd(white_vb_mask1[np.nonzero(white_vb_mask1)])
     white_vb_mask1 = white_vb_mask1[~np.isnan(white_vb_mask1)]
     white_vb_iqr1 = scipy.stats.iqr(white_vb_mask1[np.nonzero(white_vb_mask1)])
     
     white_vb_mean2 = np.nanmean(white_vb_mask2[np.nonzero(white_vb_mask2)])
     white_vb_std2 = np.nanstd(white_vb_mask2[np.nonzero(white_vb_mask2)])
     white_vb_mask2 = white_vb_mask2[~np.isnan(white_vb_mask2)]
     white_vb_iqr2 = scipy.stats.iqr(white_vb_mask2[np.nonzero(white_vb_mask2)])
     
     WMmeans[volit,:,0] = np.array([white_t1t_mean1, white_t1b_mean1, white_vb_mean1, white_taub_mean1])
     WMmeans[volit,:,1] = np.array([white_t1t_mean2, white_t1b_mean2, white_vb_mean2, white_taub_mean2])
       
     GMmeans[volit,:,0] = np.array([grey_t1t_mean1, grey_t1b_mean1, grey_vb_mean1, grey_taub_mean1])
     GMmeans[volit,:,1] = np.array([grey_t1t_mean2, grey_t1b_mean2, grey_vb_mean2, grey_taub_mean2])
      
     WMstd[volit,:,0] = np.array([white_t1t_std1, white_t1b_std1, white_vb_std1, white_taub_std1])
     WMstd[volit,:,1] = np.array([white_t1t_std2, white_t1b_std2, white_vb_std2, white_taub_std2])
      
     GMstd[volit,:,0] = np.array([grey_t1t_std1, grey_t1b_std1, grey_vb_std1, grey_taub_std1])
     GMstd[volit,:,1] = np.array([grey_t1t_std2, grey_t1b_std2, grey_vb_std2, grey_taub_std2])

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