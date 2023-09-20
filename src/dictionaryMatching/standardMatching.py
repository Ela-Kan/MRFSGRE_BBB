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
from pydicom import dcmread
import scipy.ndimage as ndimage
from scipy.signal import wiener
import scipy
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
import multiprocessing as mp
from string import digits
from sklearn.metrics import mean_squared_error
import cv2

'''
#Set plotting style 
plt.style.use('dark_background')
plt.rcParams.update({'font.size': 32})
'''
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

''' -----------------------------INPUTS--------------------------------- '''

# What format are the images in: Classic or Enhanced DICOM?
imFormat = 'E' #'C' #'E'

#Input volunteer number
volunteer_no = 21
#Other imaging factors
Regime = 'Red' #Ext
acqlen = 2000
number_of_readouts = 4
TE = '2_LARGE_IRFAT'
readout = "VDS" #'VDS' 

## TO DO: Add additional identifier here for  name of folder

#image resolutiopn  
res_x =  64; res_y = 64;

sli = 27

#Folder Path
# Image folder paths 
pathToFolder = ('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + 
    str(volunteer_no) + '/' + str(Regime) + str(acqlen) + '_' + str(readout) +
    str(number_of_readouts) + '_TE' + str(TE))
bigpath = (pathToFolder + '/DICOM/') 

# Dictionary folder
dictfolder = 'WEXandBVfreeNew'  #'SliceProfileNew' #'WEXandBVfreeNew' #'Sequence' #'InversionRecovery'

dictPath = ('/Users/emmathomson/Desktop/Local/Dictionaries/Dictionary' + dictfolder + '/')

#if you want to interpolate
interpolate = 'no '
#which filter?
filt  = 'gaussian' #'gaussian'
#for gaussian smoothing
#how much do you want to smooth - specifying sigma in pixels
# option for x-y plane (space) or along timepoints (time) 
sig_space = 0
#SIG_TIME SHOULD BE ZERO
sig_time = 1
#for median filter 
step_size = 3 #5

#Type of normalisation performed 
#Set to L2-norm (standard)
norm_type = 2 #1 #2
norm_technique = "ma_right" # "ma_right" #"ma_wrong" #"fabian" #"qiane" #"relative" #"wrong"


#What images do you want to load 
#Should normally be set to 'M'
imagetype = 'M'  #'I' #'R'  #'M' #'SUM' 

#Display masked images? 
mask_im = 'no'
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
no_entries = np.size(os.listdir(dictPath))#390000 #11250 #40500 #13500

''' ---------------------------READ IN DATA------------------------------- '''

print("Starting Data Read in:  " + str(time.strftime('%X %x %Z')))

#Read in data 
imagenames = os.listdir(bigpath)
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
        fpath = bigpath + imagenames[0]
        ds = dcmread(fpath)    
        datahold = ds.pixel_array
        irsum = ds.ImageType[-2]      
        privBlock = str(ds.private_block)
        infotxt = privBlock.splitlines()
        slopesAll = [ii for ii in infotxt if "0028, 1053" in ii]
        slope = slopesAll[0::2]
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
            fpath = bigpath + ii
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

    ''' ---------------------------INTERPOLATE?-------------------------------- '''  

    if (interpolate == 'yes'):
        print("Starting Interpolation:  " + str(time.strftime('%X %x %Z')))
        if filt == 'gaussian':
            data = ndimage.gaussian_filter(data, sigma=(sig_space, sig_space, sig_time), order=0)
        elif filt == 'median':
            data = scipy.signal.medfilt(data, kernel_size=step_size)#data = ndimage.median_filter(data, size=step_size)
        elif filt == 'weiner':
            data[np.isnan(data)] = 0
            data = wiener(data, (2, 2, 1)) 
    
        '''
        #Normalise over smoothed signals 
        for xx in range(0,res_x):
           for yy in range(0,res_y):
               data[xx,yy,:] = (data[xx,yy,:]-np.min(data[xx,yy,:]))/(np.max(data[xx,yy,:]) - np.min(data[xx,yy,:]))
        '''
       
    if norm_technique == "ma_right" or  norm_technique == "ma_wrong" or norm_technique == "relative":
        #Normalise experimental signal - normalise over signal 
        #data = -data
        for xx in range(0,res_x):
            for yy in range(0,res_y):
                mean_data[xx,yy] = np.linalg.norm((data[xx,yy,:]),norm_type)  
                data[xx,yy,:] = (data[xx,yy,:])/np.linalg.norm((data[xx,yy,:]),norm_type)#*scale

           
    if norm_technique == "fabian" or  norm_technique == "other" or norm_technique == "wrong" or norm_technique == "minmax" :
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
               
    #for ii in range(1,int(2000/(40*np.pi))):
    #    flipHold = data[:,:,int(40*np.pi)*(ii-1):int(40*np.pi)*(ii)]
    #    n = np.shape(flipHold)[2]
        '''
        for i in range(n // 2):
            flipHold[:,:,i], flipHold[:,:,n-i-1] = flipHold[:,:,n-i-1], flipHold[:,:,i]
        '''
    #    data[:,:,int(40*np.pi)*(ii-1):int(40*np.pi)*(ii)]= flipHold[:,:,::-1]
    '''           
    for ii in range(1,int(2000/(40*np.pi))):
        flipHold = data[:,:,int(2000/(40*np.pi))*(ii-1):int(2000/(40*np.pi))*(ii)]
    '''   
#data = datad/np.linalg.norm((~np.isnan(datad)))

''' --------------------------BIAS CORRECTION?------------------------------ '''   
if bias_corr == 'yes':
    #This bias is the RF homogenity 
    #Pre set bias
    print("Starting Bias Correction:  " + str(time.strftime('%X %x %Z')))
    
    rr = '*' + str(res_x) + '_bias.nii.gz' #rr = '*1.0_1.npy'
    #bias_load = nib.load(glob.glob(os.path.join(str('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + str(volunteer_no) + '/Mask_Images/Segmented/' + rr))))
    for filename in glob.glob(os.path.join(str('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + str(volunteer_no) + '/Mask_Images/Segmented/' + rr))):
    #with nib.load(os.path.join(filename)) as bias_load: 
        bias_load= nib.load(os.path.join(filename))
        bias = np.array(bias_load.dataobj).T
        #bias = (1-bias)+1
            
        bias_resized = bias #resize(np.flipud(bias), (res_x, res_y),
                              #anti_aliasing=False)
    
        bias_tiled = np.tile(bias_resized, [np.size(data,2),1,1]).T
        
        data = data*bias_tiled

''' -----------------------MASK BRAIN-------------------------- '''

print("Starting Masking:  " + str(time.strftime('%X %x %Z')))

binary_mask = 1
if mask_im == 'yes':
    
    # Old masking was too harsh 
    
    rr = '*brain_mask.nii.gz' 
    for filename in glob.glob(os.path.join(str('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + str(volunteer_no) + '/Mask_Images/' + rr))):
         
         mask_load = nib.load(filename)
         mask = np.fliplr(np.flipud(np.array(mask_load.dataobj).T))
         
         mask_resized = cv2.resize(mask, dsize=(res_x, res_y), interpolation=cv2.INTER_NEAREST)
    '''

    #New masking - from atlas 
    subAtlasFile = ('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + 
        str(volunteer_no) +'/Anatomy_Seg/outputAtlas-sub.nii.gz')
    
    subAtlasLoad = nib.load(subAtlasFile)
    subAtlas = np.array(subAtlasLoad.dataobj)
    
    atlasSliceSub = np.round((np.flipud(subAtlas[:,:,sli].T)),0)
    # smooth it to remove gaps
    atlasSliceSub = scipy.signal.medfilt(atlasSliceSub, kernel_size=7)
    atlasSliceSub[atlasSliceSub>=1] = 1
    
    mask_resized = cv2.resize(atlasSliceSub, dsize=(res_x, res_y), interpolation=cv2.INTER_NEAREST)
    '''
       
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

if seg_im == 'yes':
    seg_path = '/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + str(volunteer_no) + '/Mask_Images/Segmented/'
    
    segCSF = 0
    if volunteer_no ==6: 
        segCSF = 5
    elif volunteer_no ==3: 
        segCSF = 2
    elif volunteer_no ==8: 
        segCSF = 5
    elif volunteer_no ==21: 
        segCSF = 3
    
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
if dictfolder == 'WEXandBVfreeNew' or 'NoIRNoSliceProfile':
    ## FIX ME
    array = np.asarray(lines)

#Load lookup table  
with open(os.path.join(dictPath + "lookupTable.txt" ), 'r') as f: 
    lines = np.loadtxt(f, delimiter=",")
lookup = np.asarray(lines).T
if dictfolder == 'WEXandBVfreeNew' or 'NoIRNoSliceProfile':
    ## FIX ME
   lookup = np.asarray(lines)
lookupList = lookup.T.tolist()#list(f for f in list(lines))


norm_sims = np.linalg.norm(array, 2)
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
        #mean_sims[ii] = (np.linalg.norm(array[:,ii],norm_type))
        array[:,ii] = (array[:,ii]/(np.linalg.norm(array[:,ii],norm_type))) 
        #array[:,ii] = (array[:,ii]-np.min(array[:,ii]))/(np.max(array[:,ii]) - np.min(array[:,ii])) #* mean_data
        #array = array/np.linalg.norm(data[~np.isnan(data)])


param_est = np.zeros([res_x,res_x,6])

## FIX ME
#array = array[1:,:]
''' -----------------------------MATCHING-------------------------------- '''

print("Starting Matching:  " + str(time.strftime('%X %x %Z')))
#def matchingFunction(params):
    
#    params = tuple(params)
#    pixel_x = params[0]; pixel_y = params[1]; 

#store match quality per pixel 
snr = np.zeros([res_x, res_y])
## FIX ME 
rss = np.zeros([res_x, res_y])

#array = array[:-data_offset,:]
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

           ## FIX ME
           rss[pixel_x, pixel_y] =np.sum(np.square(normsig - array[:,max_index]))
           snr[pixel_x, pixel_y] = np.mean(normsig)/rss[pixel_x, pixel_y]
           
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
              
           #Difference in scale between dictionary and data corresponds to MO
           #Arbitrary units so reduce values to make easier to visualise
           m0 = 1#mean_data[pixel_x,pixel_y]/mean_sims[max_index] #mean_data[pixel_x,pixel_y]/ #/(10e4)
           
           param_est[pixel_x,pixel_y,:] = [t1t, t1b, res, perc, b1, m0]
'''

if __name__ == '__main__':

     ----------------------------MATCHING--------------------------------- 
    
    print("Starting Matching:  " + str(time.strftime('%X %x %Z')))
    
    #For multiprocessing use the number of available cpus 
    #THIS IS TOO MUCH: Only use 4
    pool = mp.Pool(4)
    #Generate the parameters
    coordMesh = np.meshgrid (range(res_x), range(res_y))
    coordsx = np.expand_dims(coordMesh[0], axis=2)
    coordsy = np.expand_dims(coordMesh[1], axis=2)
    coords = np.concatenate((coordsx, coordsy), axis=2)
    params = np.reshape(coords, [res_x**2, 2])
    params = params.tolist()
    #Run main function in parallel 
    #Current laptop (2021 M1 Macbook Pro) will have 8 CPUs available
    try:
        poolOut = pool.map(matchingFunction, params)
    finally:
        #Terminate and join the threads after parallelisation is done
        pool.terminate()
        pool.join()
        pool.close()
'''   
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

#Plot M0
fig, ax = plt.subplots()
fig.set_size_inches(20,15, forward=True)
plt.imshow(np.squeeze(param_est[:,:,5]), cmap='Greys_r')
cbar = plt.colorbar()
cbar.set_label(r'$M_0$')
#plt.clim(800000,2000)

snr = np.fliplr(snr)
fig, ax = plt.subplots()
fig.set_size_inches(20,15, forward=True)
plt.imshow(snr, cmap='jet', vmin = 0, vmax=50)
cbar = plt.colorbar()
cbar.set_label('SNR')


#percerror = abs(1 - error/np.max(error[np.nonzero(error)]))
snrlim = 1.6
percerror = snr*binary_mask
percerror[percerror == 0 ] = np.NaN
percerror[percerror<snrlim] = 0
percerror[percerror>snrlim] = 1
#Plot error
fig, ax = plt.subplots()
fig.set_size_inches(20,15, forward=True)
plt.imshow(np.squeeze(percerror), cmap='Set3')
plt.title('SNR limit =' + str(snrlim))
#cbar = plt.colorbar()
#cbar.set_label('SNR')


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
    for filename in glob.glob(os.path.join(str('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + str(volunteer_no) + '/Mask_Images/' + rr))):
        img = nib.load(filename)    
        aff = img.affine
        '''
        aff = np.zeros([4,4])
    
    
        aff[0,:] = [-0.4, -0.00032, -0.08804, 111.9]
        aff[1,:] = [0,     0.4,    -0.226,   -96]
        aff[2,:] = [-0.007,0.0188, 5,       0.525]
        aff[3,:] = [0,     0,       0,       1]
        '''

        if volunteer_no == 2: 
            aff[0,0] = -2.4; aff[1,1] = 2.38; aff[2,2] = 10; 
            aff[0,3] = 121; aff[1,3] = -85; aff[2,3] = 40;
            
        if volunteer_no == 3:        
            aff[0,0] = -2.4; aff[1,1] = 2.4; aff[2,2] = 10; 
            aff[0,3] = 114;  aff[1,3] = -98; aff[2,3] = 35;
    
        if volunteer_no == 5:
            aff[0,0] = -2.4; aff[1,1] = 2.38; aff[2,2] = 10; 
            aff[0,3] = 112; aff[1,3] = -85; aff[2,3] = 60;
        
        if volunteer_no == 6:
            aff[0,0] = -2.4; aff[1,1] = 2.38; aff[2,2] = 10; 
            aff[0,3] = 115; aff[1,3] = -85; aff[2,3] = 10; 

        if volunteer_no == 7:
            if res_x == 96:
                aff[0,0] = -2.28; aff[1,1] = 2.26; aff[2,2] = 10; 
                aff[0,3] = 113; aff[1,3] = -87; aff[2,3] = 60; 
                aff[2,1] = -aff[1,2]/2
            else: 
                 aff[0,0] = -1.36; aff[1,1] = 1.36; aff[2,2] = 10; 
                 aff[0,3] = 113; aff[1,3] = -87; aff[2,3] = 60; 
                 aff[2,1] = -aff[1,2]/3
                 
        if volunteer_no == 8: 
            if res_x == 96:
                aff[0,0] = -2.32; aff[1,1] = 2.30; aff[2,2] = 10; 
                aff[0,3] = 110; aff[1,3] = -94; aff[2,3] = 3; 
                aff[2,1] = -aff[1,2]/3
            else:
                aff[0,0] = -3.48; aff[1,1] = 3.46; aff[2,2] = 10; 
                aff[0,3] = 110; aff[1,3] = -94; aff[2,3] = 3; 
                aff[2,1] = -aff[1,2]/3
                
        if volunteer_no == 9: 
            if res_x == 80:
                aff[0,0] = -2.78; aff[1,1] = 2.80; aff[2,2] = 15; 
                aff[0,3] = 114; aff[1,3] = -90; aff[2,3] = 32; 
                aff[2,1] = -aff[1,2]/3
            else:
                aff[0,0] = -3.44; aff[1,1] = 3.48; aff[2,2] = 10; 
                aff[0,3] = 112; aff[1,3] = -91; aff[2,3] = 30; 
                aff[2,1] = -aff[1,2]/3
            
        if volunteer_no == 10: 
                aff[0,0] = -3.40; aff[1,1] = 3.38; aff[2,2] = 10; 
                aff[0,3] = 111; aff[1,3] = -87; aff[2,3] = 100; 
                aff[2,1] = -3*aff[1,2]/4
                
        if volunteer_no == 11: 
                aff[0,0] = -3.44; aff[1,1] = 3.48; aff[2,2] = 10; 
                aff[0,3] = 107; aff[1,3] = -83; aff[2,3] = 55; 
                aff[2,1] = -3*aff[1,2]/4
        
        if volunteer_no == 12: 
                aff[0,0] = -3.44; aff[1,1] = 3.48; aff[2,2] = 10; 
                aff[0,3] = 107; aff[1,3] = -83; aff[2,3] = 55; 
                aff[2,1] = -3*aff[1,2]/4
                
        if volunteer_no == 15: 
                aff[0,0] = -3.44; aff[1,1] = 3.48; aff[2,2] = 10; 
                aff[0,3] = 107; aff[1,3] = -89; aff[2,3] = -15; 
                aff[2,1] = -3*aff[1,2]/4
                
        if volunteer_no == 17.1: 
                aff[0,0] = -3.44; aff[1,1] = 3.48; aff[2,2] = 10; 
                aff[0,3] = 107; aff[1,3] = -82; aff[2,3] = 20; 
                aff[2,1] = -3*aff[1,2]/4
                
        if volunteer_no == 17.2: 
                aff[0,0] = -3.44; aff[1,1] = 3.48; aff[2,2] = 10; 
                aff[0,3] = 107; aff[1,3] = -82; aff[2,3] = 15; 
                aff[2,1] = -3*aff[1,2]/4        
        
        
            
        im = param_est[:,:,0]
        im = np.flipud(im).T           
        new_image = nib.Nifti1Image(im, affine=aff)
        #rotate_img = np.rot90(new_image, 180)
        filestr = 'T1_t[ms]_' + dictfolder + '.nii.gz'
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  
        
        im = param_est[:,:,1]
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        filestr = 'T1_b[ms]_' + dictfolder + '.nii.gz'
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  
        
        im = param_est[:,:,2]
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        filestr = 'tau_b[ms]_' + dictfolder + '.nii.gz'
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  
        
        im = param_est[:,:,3]
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        filestr = 'v_b[%]_' + dictfolder + '.nii.gz'
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr)) 
        
        im = param_est[:,:,4]
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        filestr = 'B1+_' + dictfolder + '.nii.gz'
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  
        
        im = param_est[:,:,5]
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        filestr = 'Relative M0[a.u.]_' + dictfolder + '.nii.gz'
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  
        
        im = snr
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        filestr = 'SNR_' + dictfolder + '.nii.gz'
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  

        im = rss
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        filestr = 'Residual_' + dictfolder + '.nii.gz'
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  


#Stop timer and print                                                    
t1 = time.time()
total = t1-t0
print("Total Time:    " +  str(total) + " seconds")  

           
'''
## RESIZE NIFTI IMAGES 
# Required before segmenting to make the T2 image the same dimensions as MRF images 

res_x = 64
res_y = res_x

rr = '*T2W*' 
for filename in glob.glob(os.path.join(str('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + str(volunteer_no) + '/Mask_Images/' + rr))):
    T2_load = nib.load(filename)     

    T2 = np.squeeze(np.array(T2_load.dataobj).T)

    T2_resized = resize(np.flipud(T2), (res_x, res_y),
                       anti_aliasing=False)

    #T2_smooth = ndimage.gaussian_filter(T2_resized, sigma=(sig_space, sig_space), order=0)

   c
    nib.save(new_image, '/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' 
             + str(volunteer_no) + '/Mask_Images/T2_w_' + str(res_x) + '.nii.gz') 


'''
'''
T1_load = nib.load('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer21/Anatomy_Seg/DICOM/DICOM_WIP_Volunteer_Anatomy_20220922170043_301.nii.gz')   
T1slice = T1_load.dataobj[:,:,29]
nib.save(new_image, '/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' 
         + str(volunteer_no) + '/Anatomy_Seg/T1regional.nii.gz')
'''
