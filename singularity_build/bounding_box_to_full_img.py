import sys
from intensity_normalization.normalize.nyul import NyulNormalize
import nibabel as nib
import numpy as np

predFilepath = sys.argv[1]
sclimbic_filepath = sys.argv[2]
predDir = sys.argv[3]
pred_out_filepath = sys.argv[4]
sclimbicDir = predDir

# split the augFilename path
predFilename = predFilepath.split("/")[-1]

subjName = predFilename.split("_")[3]
augPerformed = predFilename.split("_")[4]

## load in the images
# 3T image
pred_img = nib.load(predFilepath)
# image that contains the sclimbic mask
sclimbic_img = nib.load(sclimbic_filepath)

## get the data as a np array
pred_data = pred_img.get_fdata()
sclimbic_data = sclimbic_img.get_fdata() # previously data


## extract the mammilary bodies
# get the indices of where the mammilary bodies are 
L_mam_body_ind = np.where(sclimbic_data==843)
R_mam_body_ind = np.where(sclimbic_data==844)

# check if the mammilary bodies are not detected
if (len(L_mam_body_ind[0])==0 or len(L_mam_body_ind[1])==0 or 
    len(L_mam_body_ind[2])==0 or len(R_mam_body_ind[0])==0 or
    len(R_mam_body_ind[1])==0 or len(R_mam_body_ind[2])==0):
    warnings.warn("Mammilary body segmentation data missing for "+sclimbic_filepath)
    # check if the error file exits
    if exists(boxDir+"error.txt"):
        f = open(boxDir+"error.txt","a+")
        f.write("Mammilary bodies are missing in "+sclimbic_filepath)
        f.write("\n")
        f.close()
        sys.exit()
        # continue # skip this iteration
    else: # if the file doesn't exist we need to make it!
        f = open(boxDir+"error.txt","w+")
        f.write("Mammilary bodies are missing in "+sclimbic_filepath)
        f.write("\n")
        f.close()
        sys.exit()
        # continue # skip this iteration
    

# Combine the indices of the L and R mam bodies
x_ind = np.concatenate((L_mam_body_ind[0],R_mam_body_ind[0]))
y_ind = np.concatenate((L_mam_body_ind[1],R_mam_body_ind[1]))
z_ind = np.concatenate((L_mam_body_ind[2],R_mam_body_ind[2]))

#create a numpy array that has all zeros
mam_body_mask = np.zeros(sclimbic_data.shape)

#fill in the array with ones where there the mammilary bodies are 
mam_body_mask[x_ind,y_ind,z_ind] = 1


## Find the center of the mammillary bodies
# minimum index value
x_min = np.min(x_ind)
y_min = np.min(y_ind)
z_min = np.min(z_ind)

# maximum index value
x_max = np.max(x_ind)
y_max = np.max(y_ind)
z_max = np.max(z_ind)

# figure out the center
x_center = np.int64(np.floor(((x_max-x_min)/2) + x_min))
y_center = np.int64(np.floor(((y_max-y_min)/2) + y_min))
z_center = np.int64(np.floor(((z_max-z_min)/2) + z_min))

## Now lets extract a box with 3cm on every side
# get the limits for the x, y, and z dimension +/- 3cm aka 30 voxels since each voxel is 1mm

boxSize = 64
boxSizeHalf = np.int64(boxSize / 2)

x_lim = np.array(range(x_center-boxSizeHalf,x_center+boxSizeHalf))
y_lim = np.array(range(y_center-boxSizeHalf,y_center+boxSizeHalf))
z_lim = np.array(range(z_center-boxSizeHalf,z_center+boxSizeHalf))

masked_mam_box_64 = np.zeros((boxSize,boxSize,boxSize))
nbm_masked_mam_box_64 = np.zeros((boxSize,boxSize,boxSize))

restored_data = np.zeros(sclimbic_data.shape)

for j in range(len(x_lim)):
    for k in range(len(y_lim)):
        for l in range(len(z_lim)):
            restored_data[x_lim[j],y_lim[k],z_lim[l]] = pred_data[j,k,l]


# create a new image for the 3T and NBM data
restored_img = nib.Nifti1Image(restored_data, sclimbic_img.affine, sclimbic_img.header)

# save the 3T and NBM bounding box data 
nib.save(restored_img, pred_out_filepath)