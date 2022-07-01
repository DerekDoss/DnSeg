import sys
import nibabel as nib
from nibabel.processing import conform

inputFilePath = sys.argv[1]
outputFilePath = sys.argv[2]
img = nib.load(inputFilePath)
if (img.header.get_zooms()[0] == 1.0) and (img.header.get_zooms()[1] == 1.0) and (img.header.get_zooms()[2] == 1.0):
    # do nothing
    nib.save(img, outputFilePath)
else:
    resliced_img = conform(img, voxel_size=(1.0,1.0,1.0))
    nib.save(resliced_img, outputFilePath)