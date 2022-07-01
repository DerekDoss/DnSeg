import sys
from intensity_normalization.normalize.nyul import NyulNormalize
import nibabel as nib
import numpy as np

     
augFilename = sys.argv[1]
sclimbicDir = sys.argv[2]
boxDir = sclimbicDir
normDir = sclimbicDir

## get the filepath for the sclimbic file
# split the augFilename path
augFilenameSplit = augFilename.split("/")

# get the last element of the array, aka the filename
sclimbic_filename = augFilenameSplit[-1]

# remove the file extension
sclimbic_filename = sclimbic_filename[:-4]

# add in the suffix
sclimbic_filename = sclimbic_filename+"_sclimbic.nii"

# add in the full filepath
sclimbic_filepath = sclimbicDir+sclimbic_filename
# print("Sclimbic input filepath: " + sclimbic_filepath)


# subject name 
subjName = augFilenameSplit[-1].split("_")[0]


## get the filepath for the NBM file
# get the filename for the augmented file
augDataFilename = augFilenameSplit[-1]


## create the filepath for the 3T file with the bounding box
# get the aug filename
aug_boundingBox_filename = augFilenameSplit[-1]
# remove the file extension
# aug_boundingBox_filename = aug_boundingBox_filename[:-4]
# add in the extension
aug_boundingBox_filename = aug_boundingBox_filename.replace(".nii","_box.nii")
# create the full filepath
aug_boundingBox_filepath = boxDir+aug_boundingBox_filename
# print("3T filename with bounding box: " + aug_boundingBox_filepath)


## load in the images
# 3T image
aug_img = nib.load(augFilename)
# image that contains the sclimbic mask
sclimbic_img = nib.load(sclimbic_filepath)


## get the data as a np array
aug_data = aug_img.get_fdata()
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
    else: # if the file doesn't exist we need to make it!
        f = open(boxDir+"error.txt","w+")
        f.write("Mammilary bodies are missing in "+sclimbic_filepath)
        f.write("\n")
        f.close()
        sys.exit()
    

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
# x -> L to R
# y -> A to P
# z -> S to I

boxSize = np.array([64,64,64])
boxSizeHalf = (boxSize / 2).astype(int)

x_lim = np.array(range(x_center-boxSizeHalf[0],x_center+boxSizeHalf[0]))
y_lim = np.array(range(y_center-boxSizeHalf[1],y_center+boxSizeHalf[1]))
z_lim = np.array(range(z_center-boxSizeHalf[2],z_center+boxSizeHalf[2]))

masked_mam_box_64 = np.zeros((boxSize[0],boxSize[1],boxSize[2]))
nbm_masked_mam_box_64 = np.zeros((boxSize[0],boxSize[1],boxSize[2]))

for j in range(len(x_lim)):
    for k in range(len(y_lim)):
        for l in range(len(z_lim)):
            masked_mam_box_64[j,k,l] = aug_data[x_lim[j],y_lim[k],z_lim[l]]


# create a new image for the 3T and NBM data
masked_mam_box_64_img = nib.Nifti1Image(masked_mam_box_64, aug_img.affine, aug_img.header)

# save the 3T and NBM bounding box data 
nib.save(masked_mam_box_64_img, aug_boundingBox_filepath)