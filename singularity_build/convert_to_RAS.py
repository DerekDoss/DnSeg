import sys
import nibabel as nib

# print(sys.argv[1])
# print(sys.argv[2])

inputFilePath = sys.argv[1]
outputFilePath = sys.argv[2]

img = nib.load(inputFilePath)
canonical_img = nib.as_closest_canonical(img)
inputFilePath = inputFilePath.replace("\\", "/")
fname = inputFilePath.split("/")[-1]
nib.save(canonical_img, outputFilePath)