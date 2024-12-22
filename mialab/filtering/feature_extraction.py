"""The feature extraction module contains classes for feature extraction."""
import sys
from multiprocessing.managers import Value

import numpy as np
import pymia.filtering.filter as fltr
import SimpleITK as sitk
from numpy.ma.core import nonzero
from pymia.filtering.filter import FilterParams
from radiomics import featureextractor


class AtlasCoordinates(fltr.Filter):
    """Represents an atlas coordinates feature extractor."""

    def __init__(self):
        """Initializes a new instance of the AtlasCoordinates class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: fltr.FilterParams = None) -> sitk.Image:
        """Executes a atlas coordinates feature extractor on an image.

        Args:
            image (sitk.Image): The image.
            params (fltr.FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The atlas coordinates image
            (a vector image with 3 components, which represent the physical x, y, z coordinates in mm).

        Raises:
            ValueError: If image is not 3-D.
        """

        if image.GetDimension() != 3:
            raise ValueError('image needs to be 3-D')

        x, y, z = image.GetSize()

        # create matrix with homogenous indices in axis 3
        coords = np.zeros((x, y, z, 4))
        coords[..., 0] = np.arange(x)[:, np.newaxis, np.newaxis]
        coords[..., 1] = np.arange(y)[np.newaxis, :, np.newaxis]
        coords[..., 2] = np.arange(z)[np.newaxis, np.newaxis, :]
        coords[..., 3] = 1

        # reshape such that each voxel is one row
        lin_coords = np.reshape(coords, [coords.shape[0] * coords.shape[1] * coords.shape[2], 4])

        # generate transformation matrix
        tmp_mat = image.GetDirection() + image.GetOrigin()
        tfm = np.reshape(tmp_mat, [3, 4], order='F')
        tfm = np.vstack((tfm, [0, 0, 0, 1]))

        atlas_coords = (tfm @ np.transpose(lin_coords))[0:3, :]
        atlas_coords = np.reshape(np.transpose(atlas_coords), [z, y, x, 3], 'F')

        img_out = sitk.GetImageFromArray(atlas_coords)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'AtlasCoordinates:\n' \
            .format(self=self)


def first_order_texture_features_function(values):
    """Calculates first-order texture features.

    Args:
        values (np.array): The values to calculate the first-order texture features from.

    Returns:
        np.array: A vector containing the first-order texture features:

            - mean
            - variance
            - sigma
            - skewness
            - kurtosis
            - entropy
            - energy
            - snr
            - min
            - max
            - range
            - percentile10th
            - percentile25th
            - percentile50th
            - percentile75th
            - percentile90th
    """
    eps = sys.float_info.epsilon  # to avoid division by zero

    mean = np.mean(values)
    std = np.std(values)
    snr = mean / std if std != 0 else 0
    min_ = np.min(values)
    max_ = np.max(values)
    num_values = len(values)
    p = values / (np.sum(values) + eps)
    return np.array([mean,
                     np.var(values),  # variance
                     std,
                     np.sqrt(num_values * (num_values - 1)) / (num_values - 2) * np.sum((values - mean) ** 3) /
                     (num_values*std**3 + eps),  # adjusted Fisher-Pearson coefficient of skewness
                     np.sum((values - mean) ** 4) / (num_values * std ** 4 + eps),  # kurtosis
                     np.sum(-p * np.log2(p)),  # entropy
                     np.sum(p**2),  # energy (intensity histogram uniformity)
                     snr,
                     min_,
                     max_,
                     max_ - min_,
                     np.percentile(values, 10),
                     np.percentile(values, 25),
                     np.percentile(values, 50),
                     np.percentile(values, 75),
                     np.percentile(values, 90)
                     ])


class NeighborhoodFeatureExtractor(fltr.Filter):
    """Represents a feature extractor filter, which works on a neighborhood."""

    def __init__(self, kernel=(3, 3, 3), function_=first_order_texture_features_function):
        """Initializes a new instance of the NeighborhoodFeatureExtractor class."""
        super().__init__()
        self.neighborhood_radius = 3
        self.kernel = kernel
        self.function = function_

    def execute(self, image: sitk.Image, params: fltr.FilterParams = None) -> sitk.Image:
        """Executes a neighborhood feature extractor on an image.

        Args:
            image (sitk.Image): The image.
            params (fltr.FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The normalized image.

        Raises:
            ValueError: If image is not 3-D.
        """

        if image.GetDimension() != 3:
            raise ValueError('image needs to be 3-D')

        # test the function and get the output dimension for later reshaping
        function_output = self.function(np.array([1, 2, 3]))
        if np.isscalar(function_output):
            img_out = sitk.Image(image.GetSize(), sitk.sitkFloat32)
        elif not isinstance(function_output, np.ndarray):
            raise ValueError('function must return a scalar or a 1-D np.ndarray')
        elif function_output.ndim > 1:
            raise ValueError('function must return a scalar or a 1-D np.ndarray')
        elif function_output.shape[0] <= 1:
            raise ValueError('function must return a scalar or a 1-D np.ndarray with at least two elements')
        else:
            img_out = sitk.Image(image.GetSize(), sitk.sitkVectorFloat32, function_output.shape[0])

        img_out_arr = sitk.GetArrayFromImage(img_out)
        img_arr = sitk.GetArrayFromImage(image)
        z, y, x = img_arr.shape

        z_offset = self.kernel[2]
        y_offset = self.kernel[1]
        x_offset = self.kernel[0]
        pad = ((0, z_offset), (0, y_offset), (0, x_offset))
        img_arr_padded = np.pad(img_arr, pad, 'symmetric')

        for xx in range(x):
            for yy in range(y):
                for zz in range(z):

                    val = self.function(img_arr_padded[zz:zz + z_offset, yy:yy + y_offset, xx:xx + x_offset])
                    img_out_arr[zz, yy, xx] = val

        img_out = sitk.GetImageFromArray(img_out_arr)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'NeighborhoodFeatureExtractor:\n' \
            .format(self=self)


class RandomizedTrainingMaskGenerator:
    """Represents a training mask generator.

    A training mask is an image with intensity values 0 and 1, where 1 represents masked.
    Such a mask can be used to sample voxels for training.
    """

    @staticmethod
    def get_mask(ground_truth: sitk.Image,
                 ground_truth_labels: list,
                 label_percentages: list,
                 background_mask: sitk.Image = None) -> sitk.Image:
        """Gets a training mask.

        Args:
            ground_truth (sitk.Image): The ground truth image.
            ground_truth_labels (list of int): The ground truth labels,
                where 0=background, 1=label1, 2=label2, ..., e.g. [0, 1]
            label_percentages (list of float): The percentage of voxels of a corresponding label to extract as mask,
                e.g. [0.2, 0.2].
            background_mask (sitk.Image): A mask, where intensity 0 indicates voxels to exclude independent of the
            label.

        Returns:
            sitk.Image: The training mask.
        """

        # initialize mask
        ground_truth_array = sitk.GetArrayFromImage(ground_truth)
        mask_array = np.zeros(ground_truth_array.shape, dtype=np.uint8)

        # exclude background
        if background_mask is not None:
            background_mask_array = sitk.GetArrayFromImage(background_mask)
            background_mask_array = np.logical_not(background_mask_array)
            ground_truth_array = ground_truth_array.astype(float)  # convert to float because of np.nan
            ground_truth_array[background_mask_array] = np.nan

        for label_idx, label in enumerate(ground_truth_labels):
            indices = np.transpose(np.where(ground_truth_array == label))
            np.random.shuffle(indices)

            no_mask_items = int(indices.shape[0] * label_percentages[label_idx])

            for no in range(no_mask_items):
                x = indices[no][0]
                y = indices[no][1]
                z = indices[no][2]
                mask_array[x, y, z] = 1  # this is a masked item

        mask = sitk.GetImageFromArray(mask_array)
        mask.SetOrigin(ground_truth.GetOrigin())
        mask.SetDirection(ground_truth.GetDirection())
        mask.SetSpacing(ground_truth.GetSpacing())

        return mask

class ROIParams(FilterParams):
    """Custom FilterParams to hold ROI masks"""
    def __init__(self, roi_masks: dict):
        self.roi_masks = roi_masks

    def get_roi_masks(self):
        return self.roi_masks

class PyradiomicsROIExtractor(fltr.Filter):
    """
    Custom feature extraction filter using Pyradiomics for ROI-based feature extraction.
    """
    VALID_GLCM_FEATURES = {
        'entropy': 'DifferenceEntropy',
        'contrast': 'Contrast'
    }

    def __init__(self, enabled_feature_classes: list[str] = None, feature_params: dict=None):
        """
        Initializes a feature extractor with selected feature classes and parameters.

        Args:
            enabled_feature_classes (list[str]): A list of feature classes to enable.
            feature_params (dict): Feature-specific parameters.
        """

        super().__init__()
        self.enabled_feature_classes = enabled_feature_classes or []
        self.feature_params = feature_params or {}

        # Adapt GLCM features to valid names
        if "glcm" in self.enabled_feature_classes and "glcm" in self.feature_params:
            adapted_features = {}
            for glcm_feature in self.feature_params["glcm"]:
                feature_lower = glcm_feature.lower()
                if feature_lower in self.VALID_GLCM_FEATURES:
                    valid_name = self.VALID_GLCM_FEATURES[feature_lower]
                    adapted_features[valid_name] = True
                else:
                    raise ValueError(
                        f"Invalid GLCM feature: '{glcm_feature}'. "
                    )

            self.feature_params["glcm"] = adapted_features


    def execute(self, image: sitk.Image, params: ROIParams = None) -> dict:
        """
        Executes the feature extraction for each ROI in the input mask.

        Args:
            image (sitk.Image): The input image.
            params (ROIParams): Contains ROI masks as input parameters

        Returns:
            dict: Dictionary of feature images keyed by ROI label, feature class, and feature name
        """
        roi_masks = params.get_roi_masks()
        if roi_masks is None:
            raise ValueError("Missing 'roi_masks' in parameters for ROI-based feature extraction")

        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()

        # Enable feature classes
        for feature_class in self.enabled_feature_classes:
            extractor.enableFeatureClassByName(feature_class)
            if feature_class in self.feature_params:
                extractor.enableFeaturesByName(**{feature_class: self.feature_params[feature_class]})

        feature_images = {}

        for label, mask in roi_masks.items():
            # Extract features for current ROI
            feature_vector = extractor.execute(image, mask)

            for feature_class in self.enabled_feature_classes:
                for feature_name in self.feature_params.get(feature_class, []):
                    feature_key = f"original_{feature_class}_{feature_name}"
                    if feature_key not in feature_vector:
                        print(f"Feature {feature_key} not found for label {label}. Creating filler image.")
                        # Create a filler image (all zeros)
                        filler_array = np.zeros_like(sitk.GetArrayFromImage(mask), dtype=np.float32)
                        feature_image = sitk.GetImageFromArray(filler_array)
                        feature_image.CopyInformation(image)
                    else:
                        # Create image with feature value across ROI
                        feature_value = feature_vector[feature_key]
                        mask_array = sitk.GetArrayFromImage(mask)

                        feature_array = np.where(mask_array, feature_value, 0).astype(np.float32)

                        feature_image = sitk.GetImageFromArray(feature_array)
                        feature_image.CopyInformation(image)

                        # Save feature image
                        label = int(label)
                        feature_type_key = f"{label}_{feature_class}_{feature_name}"
                        feature_images[feature_type_key] = feature_image

        return feature_images

class PyradiomicsExtractor(fltr.Filter):
    """
    Feature extractor class that supports full-image feature extraction using Pyradiomics.
    """
    VALID_GLCM_FEATURES = {
        'entropy': 'DifferenceEntropy',
        'contrast': 'Contrast'
    }

    def __init__(self, enabled_feature_classes: list[str] = None, feature_params: dict=None):
        """
        Initializes the extractor with enable feature classes and their parameters.

        Args:
            enabled_feature_classes (list[str]): A list of feature classes to enable.
            feature_params (dict): Feature-specific parameters
        """
        super().__init__()
        self.enabled_feature_classes = enabled_feature_classes or []
        self.feature_params = feature_params or {}

        # Adapt GLCM features to valid names
        if "glcm" in self.enabled_feature_classes and "glcm" in self.feature_params:
            adapted_features = {}
            for glcm_feature in self.feature_params["glcm"]:
                feature_lower = glcm_feature.lower()
                if feature_lower in self.VALID_GLCM_FEATURES:
                    valid_name = self.VALID_GLCM_FEATURES[feature_lower]
                    adapted_features[valid_name] = True
                else:
                    raise ValueError(
                        f"Invalid GLCM feature: '{glcm_feature}'. "
                    )

            self.feature_params["glcm"] = adapted_features

    def execute(self, image: sitk.Image, mask: np.array = None, params: FilterParams = None) -> dict:
        """
        Executes the feature extraction within binary mask

        Args:
            image (sitk.Image): The input image
            mask (sitk.Image): Binary mask for feature extraction region
            params (FilterParams): Additional filter parameters

        Returns:
            dict: A dictionary of feature images keyed by feature class and feature name
        """

        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()

        # Enable specified features
        for feature_class in self.enabled_feature_classes:
            extractor.enableFeatureClassByName(feature_class)
            if feature_class in self.feature_params:
                extractor.enableFeaturesByName(**{feature_class: self.feature_params[feature_class]})

        # Make sure mask is binary with matching dimensions
        if mask is not None:
            # Make sure not empty
            mask_array = sitk.GetArrayFromImage(mask).astype(np.uint8)
            if np.count_nonzero(mask_array) == 0:
                raise ValueError("The mask is empty; cannot perform feature extraction.")

            # Check dimensions
            if mask_array.shape != sitk.GetArrayFromImage(image).shape:
                raise ValueError(
                    f"Mask shape {mask_array.shape} does not match image shape {sitk.GetArrayFromImage(image).shape}.")
            mask = sitk.GetImageFromArray(mask_array)
            mask.CopyInformation(image)


        # Perform feature extraction
        feature_vector = extractor.execute(image, mask)

        # Initialize composite image array
        feature_images = {}

        for feature_class in self.enabled_feature_classes:
            for feature_name in self.feature_params.get(feature_class, []):
                feature_key = f"original_{feature_class}_{feature_name}"

                if feature_key not in feature_vector:
                    print(
                        f"Warning: Feature 'original_{feature_class}_{feature_name}' not found in extracted features.")
                    continue

                # Add feature values to composite image array
                feature_value = feature_vector[feature_key]

                if np.isscalar(feature_value):
                    feature_image_array = np.full(sitk.GetArrayFromImage(image).shape, feature_value, dtype=np.float32)
                else:
                    mask_array = sitk.GetArrayFromImage(mask)
                    feature_image_array = np.where(mask_array > 0, feature_value, 0).astype(np.float32)

                feature_image = sitk.GetImageFromArray(feature_image_array.astype(np.float32))
                feature_image.CopyInformation(image)

                # Save feature image
                feature_type_key = f"{feature_class}_{feature_name}"
                feature_images[feature_type_key] = feature_image

                for feature_type_key, feature_image in feature_images.items():
                    print(f"{feature_type_key} Size:", feature_image.GetSize())

        return feature_images


class GlcmTextureFeatureExtractor(fltr.Filter):

    VALID_GLCM_FEATURES = ['Contrast', 'DifferenceEntropy']

    def __init__(self, glcm_feature: str):
        super().__init__()

        if glcm_feature.lower() == 'entropy':
            glcm_feature = 'DifferenceEntropy'
        elif glcm_feature.lower() == 'contrast':
            glcm_feature = 'Contrast'

        if glcm_feature not in self.VALID_GLCM_FEATURES:
            raise ValueError(f"Invalid GLCM feature: '{glcm_feature}'. Must be listed in VALID_GLCM_FEATURES")
        self.glcm_feature = glcm_feature

    def execute(self, image: sitk.Image, mask: np.array = None, params: FilterParams = None) -> sitk.Image:
        """
        Extract GLCM features.

        Args:
            image (sitk.Image): The input image.
            mask (np.array): Binary mask array.
            params (FilterParams): Parameters for feature extraction.

        Returns:
             sitk.Image: Image where each voxel encodes the extracted feature value.
        """

        # Initialize feature extractor from pyradiomics
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.enableFeatureClassByName('glcm')

        # Enable specific GLCM features based
        extractor.enableFeaturesByName(**{self.glcm_feature: True})

        # Convert NumPy mask to SimpleITK if necessary
        if isinstance(mask, np.ndarray):
            mask = sitk.GetImageFromArray(mask.astype(np.uint8))
            mask.CopyInformation(image)

        # Extract features using pyradiomics
        feature_vector = extractor.execute(image, mask)

        # Extract GLCM specific features
        feature_key = f'original_glcm_{self.glcm_feature}'
        if feature_key not in feature_vector:
            raise ValueError(f"Feature 'original_glcm_{self.glcm_feature}' not found in extracted features")

        # Assign the value for the entire feature map
        feature_value = feature_vector[feature_key]
        feature_image_array = np.full(sitk.GetArrayFromImage(image).shape, feature_value, dtype=np.float32)

        # Convert feature array back to SimpleITK image
        feature_image = sitk.GetImageFromArray(feature_image_array)
        feature_image.CopyInformation(image)

        return feature_image

class GlrlmTextureFeatureExtractor(fltr.Filter):
    def __init__(self, glrlm_feature: str):
        super().__init__()
        self.glrlm_feature = 'RunLengthNonUniformity'

    def execute(self, image: sitk.Image, mask: np.array = None, params: FilterParams = None) -> sitk.Image:
        # Initialize feature extractor
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.enableFeatureClassByName('glrlm')
        extractor.enableFeaturesByName(**{self.glrlm_feature: True})

        if isinstance(mask, np.ndarray):
            mask = sitk.GetImageFromArray(mask.astype(np.uint8))
            mask.CopyInformation(image)

        # Extract features using pyradiomics
        feature_vector = extractor.execute(image, mask)

        # Extract GLCM specific features
        feature_key = f'original_glrlm_{self.glrlm_feature}'
        if feature_key not in feature_vector:
            raise ValueError(f"Feature 'original_glrlm_{self.glrlm_feature}' not found in extracted features")

        # Assign the value for the entire feature map
        feature_value = feature_vector[feature_key]
        feature_image_array = np.full(sitk.GetArrayFromImage(image).shape, feature_value, dtype=np.float32)

        # Convert feature array back to SimpleITK image
        feature_image = sitk.GetImageFromArray(feature_image_array)
        feature_image.CopyInformation(image)

        return feature_image



