#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""------------------------------------------------------------------------
Standard inner product MRF matching with B1 prematched, smoothed, and the 
remaining parameters matched 

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
from scipy import interpolate
from scipy import signal
import cv2

'''
#Set plotting style 
plt.style.use('dark_background')
plt.rcParams.update({'font.size': 32})
'''
#lt.style.use('bmh')
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

# What format are th images in: Classic or Enhanced DICOM?
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

#image resolution  
res_x =  64; res_y = 64;
flip_T2 = False
flip_mask = False
flip_seg = False

#Folder Path
# Image folder paths 
pathToFolder = ('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + 
    str(volunteer_no) + '/' + str(Regime) + str(acqlen) + '_' + str(readout) +
    str(number_of_readouts) + '_TE' + str(TE))
bigpath = (pathToFolder + '/DICOM/') 

# Dictionary folder
dictfolder =   'WEXandBVfreeNew' #'WEXandBVfreeNew'  #'SliceProfileNew' #'Sequence' #'InversionRecovery'

dictPath = ('/Users/emmathomson/Desktop/Local/Dictionaries/Dictionary' + dictfolder + '/')

#snr0file = (pathToFolder + '/Maps/SNR0.nii.gz') 

b1smooth = 'grid' #'polynomial' #'gaussian' #'spline'
#which filter?
filt  = 'gaussian' #'gaussian'
#for gaussian smoothing
#how much do you want to smooth - specifying sigma in pixels
# option for x-y plane (space) or along timepoints (time) 
sig_space = 1
#for median filter 
step_size = 0 #5cd 

b1_dict_step = 2

denoise = False
dim = 'S'
filt = 'G'


#Type of normalisation performed 
#Set to L2-norm (standard)
norm_type = 2 #1 #2
norm_technique = "ma_right" # "ma_right" #"ma_wrong" #"fabian" #"qiane" #"relative" #"wrong"

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
#for gaussian filter
sig = 0.5
#for median filter 
step_size = 9 #5
#for anisotropic diffussion
iters = 3
kap = 50
gam = 0.25
#for bilateral filtering 
d = 3
sigColor = 25
sigSpace = 25

if filt == 'MF':
    imName = bigpath + 'IM_0001_MF_' + dim + '_' + str(step_size) + '.nii'
elif filt == 'G':
    imName = bigpath + 'IM_0001_G_' + dim + '_' + str(sig) + '.nii'
elif filt == 'AD':
    imName = bigpath + 'IM_0001_AD_'+ dim + '_' + str(iters) + '_' + str(kap) + '_' + str(gam) +  '.nii'
elif filt == 'BF':
    imName = bigpath + 'IM_0001_BF_'+ dim + '_' + str(d) + '_' + str(sigColor) + '_' + str(sigSpace) +  '.nii'
elif filt == 'PCA':
    imName = bigpath + 'IM_0001_PCA.nii'


#Number of FA is less than acquisition length because we want to get rid of initial values 
#initial_bin = 1
#no_of_FA = acqlen-initial_bin
#data_offset = 1
#Number of entries in dictionary (to set array sizes)
no_entries = np.size(os.listdir(dictPath))#390000 #11250 #40500 #13500

''' ---------------------------READ IN DATA------------------------------- '''

print("Starting Data Read in:  " + str(time.strftime('%X %x %Z')))

if denoise is False:
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
    
        data1 = np.zeros(np.shape(data))
        if norm_technique == "ma_right" or  norm_technique == "ma_wrong" or norm_technique == "relative":
            #Normalise experimental signal - normalise over signal 
            #data = -data
            for xx in range(0,res_x):
                for yy in range(0,res_y):
                    mean_data[xx,yy] = np.linalg.norm((data[xx,yy,:]),norm_type)  
                    data[xx,yy,:] = (data[xx,yy,:])/np.linalg.norm((data[xx,yy,:]),norm_type)#*scale
        
        if norm_technique == "mixed":
             for xx in range(0,res_x):
                for yy in range(0,res_y):
                    mean_data[xx,yy] = np.linalg.norm((data[xx,yy,:]),norm_type)  
                    data[xx,yy,:] = (data[xx,yy,:])/np.linalg.norm((data[xx,yy,:]),norm_type)#*scale
                    data1[xx,yy,:] = (data[xx,yy,:]/(data[xx,yy,0]))*1000#*scale
            
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
else: 
    mean_data = np.zeros([res_x,res_y])
    data = np.squeeze(np.rot90(nib.load(imName).dataobj))

    for xx in range(0,res_x):
        for yy in range(0,res_y):
            mean_data[xx,yy] = np.linalg.norm((data[xx,yy,:]),norm_type)  
            data[xx,yy,:] = (data[xx,yy,:])/np.linalg.norm((data[xx,yy,:]),norm_type)

if flip_T2 is False: 
    data = np.fliplr(data) 
        
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
    rr = '*brain_mask.nii.gz' 
    for filename in glob.glob(os.path.join(str('/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + str(volunteer_no) + '/Mask_Images/' + rr))):
        
        mask_load = nib.load(filename)
        mask = (np.flipud(np.array(mask_load.dataobj).T))
        if flip_mask is True:
            mask = np.fliplr(mask)

        '''
        if volunteer_no == 29.1:
            mask = np.fliplr(mask)
         '''
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


seg_path = '/Users/emmathomson/Dropbox/Scanning_Documents/Scanning_Data/Volunteer' + str(volunteer_no) + '/Mask_Images/Segmented/'

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
try:
    grey_image = skimage.color.rgb2grey(segCSF)
except:
    grey_image = segCSF
if flip_seg is True:
    segCSF = np.fliplr(segCSF)
histogram, bin_edges = np.histogram(grey_image, bins=256, range=(0.0, 1.0))
t = np.max(segCSF)*0.25
binary_seg_mask = segCSF > t
binary_seg_mask_CSF = binary_seg_mask.astype('uint8')

binary_mask_seg= np.float64(binary_mask-binary_seg_mask_CSF*binary_mask + (1-binary_mask))
'''
if flip_seg is True:
        binary_mask_seg = np.fliplr(binary_mask_seg)
        
'''    
#Use MF to remove lone voxels segmented out - only interested in ventricles    
#binary_mask_seg = signal.medfilt2d(binary_mask_seg, kernel_size=3)
binary_mask_seg[binary_mask_seg == 0] = np.NaN       
#Use MF to remove lone voxels segmented out - only interested in ventricles 
#binary_mask_seg = signal.medfilt2d(binary_mask_seg, kernel_size=3)
#binary_mask_seg[np.isnan(binary_mask_seg)] = 0


#Pad array to ensure that the ventricles and surrounding distortions are fully removed 
#binary_mask_seg = np.float64(1-binary_mask_seg)
#binary_mask_seg[binary_mask_seg == 0] = np.NaN
gradx = np.gradient(binary_mask_seg)[0] 
grady = np.gradient(binary_mask_seg)[1]
gradtot = abs(gradx) + abs(grady) #abs(grad[0]) + abs(grad[1])
           
gradtot[gradtot > 0] = 1
#gradtot[np.isnan(gradtot)] = 0
gradtot = abs(1-gradtot)

#binary_mask_seg[np.isnan(binary_mask_seg)] = 1
binary_mask_seg = abs(gradtot + binary_mask_seg)


'''
binary_mask_seg[binary_mask_seg>0] = np.NaN
binary_mask_seg[binary_mask_seg == 0] = 1
binary_mask_seg[np.isnan(binary_mask_seg)] = 0
binary_mask_seg = (1- binary_mask_seg)
'''

binary_mask_seg[binary_mask_seg>2] = 0 
binary_mask_seg[binary_mask_seg == 2] = 1

'''
if volunteer_no == 29.1:
    binary_mask = np.fliplr(binary_mask)
binary_mask_tiled = np.tile(np.expand_dims(binary_mask,2),acqlen)
#Multiply to mask data
#data = data * binary_mask_tiled   
''' 

''' --------------------------READ IN SIMS------------------------------ '''
if 'array' in globals():
    array = globals()['array']
else:
    print("Starting Dictionary Read in:  " + str(time.strftime('%X %x %Z')))
    
    #Open empy arrays for dictionary signals
    mean_sims = np.zeros([no_entries])
    
    #Loading all dictionary signals into array    
    with open(os.path.join(dictPath + "dictionary.txt" ), 'r') as f:
        lines = np.loadtxt(f, delimiter=",")
    if dictfolder == 'WEXandBVfreeNew':
        array = np.asarray(lines)
    else:
        array = np.asarray(lines).T
    #array = array.astype(int)
    
    #Load lookup table  
    with open(os.path.join(dictPath + "lookupTable.txt" ), 'r') as f:
        lines = np.loadtxt(f, delimiter=",")
    if dictfolder == 'WEXandBVfreeNew':
        lookup = np.asarray(lines)
    else:
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
     
    if norm_technique == 'mixed':
        
        array1 = 2*array/2
        for ii in range(0,np.size(array,1)):
            mean_sims[ii] = (np.linalg.norm(array[:,ii],norm_type))
            array[:,ii] = (array[:,ii]/(np.linalg.norm(array[:,ii],norm_type))) 
            #array1[:,ii] = (array1[:,ii]/array1[0,ii])
            #array[:,ii] = (array[:,ii]-np.min(array[:,ii]))/(np.max(array[:,ii]) - np.min(array[:,ii])) #* mean_data
        #arr ay = array/np.linalg.norm(data[~np.isnan(data)])
        
    #array = array/np.expand_dims(scale,axis=1)

param_est = np.zeros([res_x,res_x,6])
norm_sims = np.linalg.norm(array, 2, axis=0)

''' -----------------------------MATCHING-------------------------------- '''

print("Starting Matching 1:  " + str(time.strftime('%X %x %Z')))
#def matchingFunction(params):
    
#    params = tuple(params)
#    pixel_x = params[0]; pixel_y = params[1]; 

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
           
           ##FIX ME
           #Doesnt work as normal with dot product so use RMSE 
           #RMSEerror = ((np.abs(np.tile(np.expand_dims(normsig,1), [1,np.shape(array1)[1]]) - array1))**2)/acqlen
           #RMSEerror = np.sqrt(np.sum(RMSEerror,0))
           
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
              
           #Difference in scale between dictionary and data corresponds to MO
           #Arbitrary units so reduce values to make easier to visualise
           m0 = mean_data[pixel_x,pixel_y]/mean_sims[max_index] #mean_data[pixel_x,pixel_y]/ #/(10e4)
           
           param_est[pixel_x,pixel_y,:] = [t1t, t1b, res, perc, b1, m0]

'''--------------------------B1 SMOOTHING--------------------------------- '''

#Gaussian smooth
#b1HoldMatch = ndimage.gaussian_filter(param_est[:,:,4], sigma=(sig_space, sig_space), order=0)

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

#Z[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Z[~mask])

if b1smooth == 'polynomial':
     
    Z[Z>1.1] = np.NaN
    Z[Z<0.7] = 0.7

    # We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
    x, y = X.ravel(), Y.ravel()
    # Maximum order of polynomial term in the basis.
    max_order = 6
    basis = get_basis(x, y, max_order)
    # Linear, least-squares fit.
    A = np.vstack(basis).T
    b = Z.ravel()
    A = A[~np.isnan(b)]
    b = b[~np.isnan(b)]
    c, r, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    print('Fitted parameters:')
    print(c)
    
    # Calculate the fitted surface from the coefficients, c.
    fit = np.sum(c[:, None, None] * np.array(get_basis(X, Y, max_order))
                    .reshape(len(basis), *X.shape), axis=0)
    #correct any edge cases by setting to the lowest value
    fit = fit*binary_mask
    fit[fit < 0.7] = 0.7
    fit[fit > 1.2] = 1.2
    
    rms = np.sqrt(np.nanmean((Z - fit)**2))
    print('RMS residual =', rms)
    
elif b1smooth == 'spline':
    
    #make it zero mean
    Z[Z<0.7] = 0.7
    Zbar = np.mean(Z)
    #Z = Z - Zbar
    #mask = np.isnan(Z)
    #Z[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Z[~mask])
    
    #x_edges, y_edges = np.mgrid[-1:2:33j, -1:2:33j]
    #x = x_edges[:-1, :-1] + np.diff(x_edges[:2, 0])[0] / 2.
    #y = y_edges[:-1, :-1] + np.diff(y_edges[0, :2])[0] / 2.
    # The two-dimensional domain of the fit.
    xxmin, xxmax, nx = 5, 59, 27
    yymin, yymax, ny = 5, 59, 27
    xx, yy = np.linspace(xxmin, xxmax, nx), np.linspace(yymin, yymax, ny)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = Z[5:59, 5:59]
    ZZ = ZZ[1::2,1::2]

    plt.figure()
    lims = dict(cmap='YlGnBu_r', vmin=0.6, vmax=1.2)
    plt.pcolormesh(X, Y, Z, shading='flat', **lims)
    plt.plot(XX,YY,'or')
    plt.colorbar()
    plt.title("B1+ Sampling")
    plt.show()
    
    plt.figure()
    lims = dict(cmap='YlGnBu_r', vmin=0.6, vmax=1.2)
    plt.pcolormesh(XX, YY, ZZ, shading='flat', **lims)
    plt.colorbar()
    plt.title("B1+ Sampling")
    plt.show()

    tck = interpolate.bisplrep(XX, YY, ZZ, s=0)
    fit = interpolate.bisplev(x, y, tck)
        
    plt.figure()
    plt.pcolormesh(y, x, fit, shading='flat', **lims)
    plt.colorbar()
    plt.title("Interpolated B1+")
    plt.show()
        
    fit = np.resize(fit, [res_x, res_y])
    
elif b1smooth == 'gaussian':
    
    #Z[Z<0.7] = np.NaN
    
    fit = ndimage.gaussian_filter(Z, sigma=(sig_space, sig_space), order=0)
    fit  = fit*binary_mask
    fit[fit<0.7] = 0.7
    
elif b1smooth == 'median':
    
    fit = signal.medfilt2d(Z, kernel_size=9)
    fit  = fit*binary_mask
    fit[fit<0.7] = 0.7
    
if b1smooth == 'grid':
    
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

    rms = np.sqrt(np.nanmean((Z - fit)**2))
    print('RMS residual =', rms)
    

    fit = ndimage.gaussian_filter(fit, sigma=(1.5, 1.5), order=0)
    fit  = fit*binary_mask
    fit[fit<0.7] = 0.7


elif b1smooth == 'none':
 
    fit  = Z*binary_mask
    fit[fit<0.7] = 0.7

## FIX ME 
# Plot the 3D figure of the fitted function and the residuals.

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
#def matchingFunction(params):
    
#    params = tuple(params)
#    pixel_x = params[0]; pixel_y = params[1]; 

get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]

#store match quality per pixel 
error = np.zeros([res_x, res_y])
rss = np.zeros([res_x, res_y])
snr = np.zeros([res_x, res_y])
#snrRel = np.zeros([res_x, res_y])
#snr0 = np.squeeze(np.rot90(nib.load(snr0file).dataobj))


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
              
           #Difference in scale between dictionary and data corresponds to MO
           #Arbitrary units so reduce values to make easier to visualise
           m0 = 0#mean_data[pixel_x,pixel_y]/mean_sims[max_index] #mean_data[pixel_x,pixel_y]/ #/(10e4)
           
           param_est[pixel_x,pixel_y,:] = [t1t, t1b, res, perc, b1, m0]

#snrRel = snr - snr0            
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

#Plot M0
fig, ax = plt.subplots()
fig.set_size_inches(20,15, forward=True)
plt.imshow(np.squeeze(param_est[:,:,5]), cmap='Greys_r')
cbar = plt.colorbar()
cbar.set_label(r'$M_0$')
#plt.clim(800000,2000)

percerror = abs(1 - error/np.max(error[np.nonzero(error)]))
percerror = percerror*binary_mask
#Plot error
fig, ax = plt.subplots()
fig.set_size_inches(20,15, forward=True)
plt.imshow(np.squeeze(percerror), cmap='hot_r')
cbar = plt.colorbar()
cbar.set_label('Relative Error')
plt.clim(0,1)

'''
fig, ax = plt.subplots()
fig.set_size_inches(20,15, forward=True)
limitsSNR= np.max(snrRel)
plt.imshow(snrRel, cmap='RdBu', vmin = -40, vmax=40)
cbar = plt.colorbar()
cbar.set_label('Relative SNR')
'''
#snr = np.fliplr(snr)
#percerror = abs(1 - error/np.max(error[np.nonzero(error)]))
snrlim = 1.6
percerror = 2*snr/2
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
        if denoise is True: 
            if filt == 'G':
                filestr = 'B1first_T1_t[ms]_' + dictfolder + '_G_' + dim + '_' + str(sig) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'B1first_T1_t[ms]_' + dictfolder + '_MF_' + dim + '_' + str(step_size) + '.nii.gz'
        else: 
             filestr = 'B1first_T1_t[ms]_' + dictfolder + '.nii.gz' 
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  
        
        im = param_est[:,:,1]
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        if denoise is True: 
            if filt == 'G':
                filestr = 'B1first_T1_b[ms]_' + dictfolder + '_G_' + dim + '_' + str(sig) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'B1first_T1_b[ms]_' + dictfolder + '_MF_' + dim + '_' + str(step_size) + '.nii.gz'
        else: 
            filestr = 'B1first_T1_b[ms]_' + dictfolder + '.nii.gz' 
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  
        
        im = param_est[:,:,2]
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        if denoise is True: 
            if filt == 'G':
                filestr = 'B1first_tau_b[ms]_' + dictfolder + '_G_' + dim + '_' + str(sig) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'B1first_tau_b[ms]_' + dictfolder + '_MF_' + dim + '_' + str(step_size) + '.nii.gz'
        else: 
            filestr = 'B1first_tau_b[ms]_' + dictfolder + '.nii.gz' 
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  
        
        im = param_est[:,:,3]
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        if denoise is True:
            if filt == 'G':
                filestr = 'B1first_v_b[%]_' + dictfolder + '_G_' + dim + '_' + str(sig) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'B1first_v_b[%]_' + dictfolder + '_MF_' + dim + '_' + str(step_size) + '.nii.gz'
        else: 
            filestr = 'B1first_v_b[%]_' + dictfolder + '.nii.gz' 
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr)) 
        
        im = param_est[:,:,4]
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        if denoise is True: 
            if filt == 'G':
                filestr = 'B1first_B1+_' + dictfolder + '_G_' + dim + '_' + str(sig) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'B1first_B1+_' + dictfolder + '_MF_' + dim + '_' + str(step_size) + '.nii.gz'
        else:
            filestr = 'B1first_B1+_' + dictfolder + '.nii.gz' 
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  
        
        im = param_est[:,:,5]
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        if denoise is True: 
            if filt == 'G':
                filestr = 'B1first_Relative M0[a.u.]_' + dictfolder + '_G_' + dim + '_' + str(sig) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'B1first_Relative M0[a.u.]_' + dictfolder + '_MF_' + dim + '_' + str(step_size) + '.nii.gz'
        else: 
            filestr = 'B1first_Relative M0[a.u.]_' + dictfolder + '.nii.gz' 
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  
        
        im = snr
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        if denoise is True:
            if filt == 'G':
                filestr = 'B1first_SNR_' + dictfolder + '_G_' + dim + '_' + str(sig) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'B1first_SNR_' + dictfolder + '_MF_' + dim + '_' + str(step_size) + '.nii.gz'
        else: 
            filestr = 'B1first_SNR_' + dictfolder + '.nii.gz' 
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  
        
        im = rss
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        if denoise is True: 
            if filt == 'G':
                filestr = 'B1first_Residual_' + dictfolder + '_G_' + dim + '_' + str(sig) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'B1first_Residual_' + dictfolder + '_MF_' + dim + '_' + str(step_size) + '.nii.gz' 
        else: 
            filestr = 'B1first_Residual_' + dictfolder + '.nii.gz' 
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  
        
        im = error
        im = np.flipud(im).T
        new_image = nib.Nifti1Image(im, affine=aff)
        if denoise is True: 
            if filt == 'G':
                filestr = 'B1first_Relative error_' + dictfolder + '_G_' + dim + '_' + str(sig) +  '.nii.gz'
            elif filt == 'MF': 
                filestr = 'B1first_Relative error_' + dictfolder + '_MF_' + dim + '_' + str(step_size) + '.nii.gz' 
        else:
            filestr = 'B1first_Relative error_' + dictfolder + '.nii.gz' 
        #rotate_img = np.rot90(new_image, 180)
        nib.save(new_image, os.path.join(pathToFolder, 'Maps', filestr))  


#Stop timer and print                                                    
t1 = time.time()
total = t1-t0
print("Total Time:    " +  str(total) + " seconds")  

           