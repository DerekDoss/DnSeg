import sys
from intensity_normalization.normalize.nyul import NyulNormalize
import nibabel as nib

# print("starting python")
# print(sys.argv[1])
# print(sys.argv[2])

standardHistogramPath = sys.argv[1]
inputFilePath = sys.argv[2]
normFilePath = sys.argv[3]


nyul_normalizer = NyulNormalize()
nyul_normalizer.load_standard_histogram(standardHistogramPath)


img = nib.load(inputFilePath)
normalized = nyul_normalizer(img)
nib.save(normalized, normFilePath)