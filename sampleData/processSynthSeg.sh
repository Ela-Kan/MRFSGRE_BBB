#!/bin/bash

# N.B. BEFORE RUNNING COPY THIS FILE INTO THE "VOLUNTEERXX.X" FOLDER
# N.B. ALSO DELETE FROM FOLDER AFTER USE SO THAT NUMOUROUS FILES DONT GET CONFUSING 

## Convert to nifti 
#T1 (anatomy) - to send to volunteer 

dcm2niix -z y -f T1_anatomy -o "$(dirname "$0")"/Anatomy/ "$(dirname "$0")"/Anatomy/DICOM/

#T1 (for segmentation)

dcm2niix -z y -f T1_seg -o "$(dirname "$0")"/Anatomy_Seg/ "$(dirname "$0")"/Anatomy_Seg/DICOM/

#T2 (for segmentation)

dcm2niix -z y -f T2 -o "$(dirname "$0")"/T2/ "$(dirname "$0")"/T2/DICOM/

#Copy T2 into Mask-Images 

cp "$(dirname "$0")"/T2/T2.nii.gz "$(dirname "$0")"/Mask_Images/T2.nii.gz 

#Register T1 with the T1_seg file 
#Isolate central slice in T1_seg
PYCMD=$(cat <<EOF

import os
import nibabel as nib

pathtofold = './Anatomy_Seg/'
pathtoT1 = (pathtofold + 'T1_seg.nii.gz')

t1image = nib.load(pathtoT1)._dataobj
t1image = t1image[:,:,27]
new_image = nib.Nifti1Image(t1image, affine=nib.load(pathtoT1).affine)

nib.save(new_image, os.path.join(pathtofold, 'T1_slice.nii.gz'))  
EOF
)

python3 -c "$PYCMD"

#register T2 to T1

flirt -2D -in "$(dirname "$0")"/Mask_Images/T2.nii.gz -ref "$(dirname "$0")"/Anatomy_Seg/T1_slice.nii.gz -out "$(dirname "$0")"/Mask_Images/T2_reg.nii.gz

#brain mask from T2 

# f: fractional intensity threshold, m: binary mask
bet "$(dirname "$0")"/Mask_Images/T2_reg.nii.gz  "$(dirname "$0")"/Mask_Images/T2_brain.nii.gz -m -f 0.4

#segmentation from T2 

mkdir "$(dirname "$0")"/Mask_Images/Segmented/
cp "$(dirname "$0")"/Mask_Images/T2_reg.nii.gz "$(dirname "$0")"/Mask_Images/Segmented/T2.nii.gz 

#t: 1-t1, 2-t2, 3-pd, n:number of classes, g:separate image per class, b:bias field
fast -t 2 -n 6 -g -b "$(dirname "$0")"/Mask_Images/Segmented/T2.nii.gz

#copy atlas you want into the anatomy_seg folder
cp /Users/emmathomson/Dropbox/PhD\ Work/Scanning_Documents/Segmentation_Files/HarvardOxford-sub-maxprob-thr50-1mm.nii.gz "$(dirname "$0")"/Anatomy_Seg/atlas_sub.nii.gz
cp /Users/emmathomson/Dropbox/PhD\ Work/Scanning_Documents/Segmentation_Files/HarvardOxford-cort-maxprob-thr50-1mm.nii.gz "$(dirname "$0")"/Anatomy_Seg/atlas_cort.nii.gz
cp /Users/emmathomson/Dropbox/PhD\ Work/Scanning_Documents/Segmentation_Files/MNI152_T1_1mm.nii.gz "$(dirname "$0")"/Anatomy_Seg/T1_reg.nii.gz

reg_aladin -ref "$(dirname "$0")"/Anatomy_Seg/T1_seg.nii.gz -flo "$(dirname "$0")"/Anatomy_Seg/T1_reg.nii.gz  -res "$(dirname "$0")"/Anatomy_Seg/T1_reg_out_aff.nii.gz -aff "$(dirname "$0")"/Anatomy_Seg/T1_reg_out_aff.txt

reg_f3d -ref "$(dirname "$0")"/Anatomy_Seg/T1_seg.nii.gz -flo "$(dirname "$0")"/Anatomy_Seg/T1_reg.nii.gz -res "$(dirname "$0")"/Anatomy_Seg/T1_reg_out_aff.nii.gz -aff "$(dirname "$0")"/Anatomy_Seg/T1_reg_out_aff.txt -cpp "$(dirname "$0")"/Anatomy_Seg/T1_reg_out_cpp.nii.gz

reg_resample -ref "$(dirname "$0")"/Anatomy_Seg/T1_seg.nii.gz -flo "$(dirname "$0")"/Anatomy_Seg/atlas_sub.nii.gz -res "$(dirname "$0")"/Anatomy_Seg/outputAtlas-sub.nii.gz -cpp "$(dirname "$0")"/Anatomy_Seg/T1_reg_out_cpp.nii.gz -inter 0
reg_resample -ref "$(dirname "$0")"/Anatomy_Seg/T1_seg.nii.gz -flo "$(dirname "$0")"/Anatomy_Seg/atlas_cort.nii.gz -res "$(dirname "$0")"/Anatomy_Seg/outputAtlas-cort.nii.gz -cpp "$(dirname "$0")"/Anatomy_Seg/T1_reg_out_cpp.nii.gz -inter 0

#register T1
flirt -interp spline -in "$(dirname "$0")"/Anatomy_Seg/T1_reg.nii.gz -ref "$(dirname "$0")"/Anatomy_Seg/T1_seg.nii.gz  -out "$(dirname "$0")"/Anatomy_Seg/T1_reg_out.nii.gz -omat "$(dirname "$0")"/Anatomy_Seg/T1_reg_mat.mat -v 
fnirt --ref="$(dirname "$0")"/Anatomy_Seg/T1_seg.nii.gz --in="$(dirname "$0")"/Anatomy_Seg/T1_reg.nii.gz --iout="$(dirname "$0")"/Anatomy_Seg/outputT1.nii.gz --verbose --aff="$(dirname "$0")"/Anatomy_Seg/T1_reg_mat.mat

#apply warp to atlases
applywarp -i "$(dirname "$0")"/Anatomy_Seg/atlas_sub.nii.gz -o "$(dirname "$0")"/Anatomy_Seg/outputAtlas-sub.nii.gz  -r "$(dirname "$0")"/Anatomy_Seg/T1_seg.nii.gz -w "$(dirname "$0")"/Anatomy_Seg/T1_reg_warpcoef.nii.gz
applywarp -i "$(dirname "$0")"/Anatomy_Seg/atlas_cort.nii.gz -o "$(dirname "$0")"/Anatomy_Seg/outputAtlas-cort.nii.gz  -r "$(dirname "$0")"/Anatomy_Seg/T1_seg.nii.gz -w "$(dirname "$0")"/Anatomy_Seg/T1_reg_warpcoef.nii.gz


#Run SynthSeg for parcellation of the T1 seg scan
python  /Users/emmathomson/Desktop/Local/Source_Code/SynthSeg/scripts/commands/SynthSeg_predict.py --i "$(dirname "$0")"/Anatomy_Seg/T1_seg.nii.gz --o "$(dirname "$0")"/Anatomy_Seg/outputAtlas.nii.gz --parc --resample "$(dirname "$0")"/Anatomy_Seg/outputAtlas-subRes.nii.gz --fast

#register T1
flirt -interp spline -in "$(dirname "$0")"/Anatomy_Seg/outputAtlas-subRes.nii.gz -ref "$(dirname "$0")"/Anatomy_Seg/T1_seg  -out "$(dirname "$0")"/Anatomy_Seg/T1_reg_out -omat "$(dirname "$0")"/Anatomy_Seg/T1_reg_mat.mat -v 
flirt -interp nearestneighbour -in "$(dirname "$0")"/Anatomy_Seg/outputAtlas.nii.gz -ref "$(dirname "$0")"/Anatomy_Seg/T1_seg -out "$(dirname "$0")"/Anatomy_Seg/outputAtlas-sub -init "$(dirname "$0")"/Anatomy_Seg/T1_reg_mat.mat -applyxfm 

#apply warp to atlases
#applywarp -r "$(dirname "$0")"/Anatomy_Seg/T1_seg.nii.gz -i "$(dirname "$0")"/Anatomy_Seg/atlas_sub.nii.gz -o "$(dirname "$0")"/Anatomy_Seg/outputAtlas-sub.nii.gz  -r "$(dirname "$0")"/Anatomy_Seg/T1_seg.nii.gz -w "$(dirname "$0")"/Anatomy_Seg/T1_reg_warpcoef.nii.gz --interp=nn
#applywarp -r "$(dirname "$0")"/Anatomy_Seg/T1_seg.nii.gz -i "$(dirname "$0")"/Anatomy_Seg/atlas_cort2.nii.gz -o "$(dirname "$0")"/Anatomy_Seg/outputAtlas-cort.nii.gz  -r "$(dirname "$0")"/Anatomy_Seg/T1_seg.nii.gz -w "$(dirname "$0")"/Anatomy_Seg/T1_reg_warpcoef.nii.gz --interp=nn
