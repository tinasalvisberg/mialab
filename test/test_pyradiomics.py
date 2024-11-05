from radiomics import featureextractor
import numpy as np
import SimpleITK as sitk

# Simple test image (cube with uniform intensity)
image_data = np.ones((50, 50, 50))
image = sitk.GetImageFromArray(image_data)

# Simple binary mask with label
mask_data = np.zeros((50, 50, 50))
mask_data[20:30, 20:30, 20:30] = 1
mask = sitk.GetImageFromArray(mask_data)

# Initialize the feature extractor with default settings
extractor = featureextractor.RadiomicsFeatureExtractor()

# Extract features from the test image and mask
features = extractor.execute(image, mask)

# Print the result
print("Feature extraction successful, features extracted:")
print(features)
