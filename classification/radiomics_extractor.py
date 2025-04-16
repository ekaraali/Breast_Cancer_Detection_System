import os
import six
import SimpleITK as sitk
import pandas as pd
from radiomics import featureextractor
import matplotlib.pyplot as plt

class RadiomicsFeatureExtractor:
    """
    RadiomicsFeatureExtractor encapsulates the feature extraction process using PyRadiomics.
    It extracts radiomics features from an image and its corresponding mask.
    """
    def __init__(self, params=None):
        """
        Args:
            params (dict, optional): Parameters for the PyRadiomics feature extractor.
        """
        if params is None:
            params = {}  # default parameters can be provided here
        self.extractor = featureextractor.RadiomicsFeatureExtractor(**params)

    @staticmethod
    def display_images(image_sitk, mask_sitk):
        """
        Display the original image and mask side-by-side for visual verification.
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(sitk.GetArrayFromImage(image_sitk), cmap='gray')
        axes[0].set_title('Original Image')
        axes[1].imshow(sitk.GetArrayFromImage(mask_sitk), cmap='gray')
        axes[1].set_title('Mask Image')
        plt.show()

    def extract_features_from_folder(self, folder, label):
        """
        Extract features from all image/mask pairs in a folder.

        Args:
            folder (str): Directory path containing images and their corresponding masks.
            label (int): Label to assign (e.g. 0 for benign, 1 for malignant).

        Returns:
            pd.DataFrame: DataFrame containing extracted features with the given label.
        """
        features_dataframe = pd.DataFrame()

        for filename in os.listdir(folder):
            if filename.endswith("_mask.png"):
                image_path = os.path.join(folder, filename.replace('_mask', ''))
                mask_path = os.path.join(folder, filename)

                if os.path.exists(image_path):
                    # Read image and mask
                    mask_sitk = sitk.ReadImage(mask_path, sitk.sitkUInt8)
                    image_sitk = sitk.ReadImage(image_path, sitk.sitkUInt8)

                    # Convert to 3D (single-slice volumes)
                    image_sitk = sitk.JoinSeries(image_sitk)
                    mask_sitk = sitk.JoinSeries(mask_sitk)

                    # Ensure mask is binary
                    mask_sitk = sitk.BinaryThreshold(mask_sitk, lowerThreshold=1, upperThreshold=255,
                                                     insideValue=1, outsideValue=0)

                    # Uncomment the next line to visualize images for debugging:
                    # self.display_images(image_sitk, mask_sitk)

                    # Extract features using PyRadiomics
                    result = self.extractor.execute(image_sitk, mask_sitk)

                    # Keep only features starting with "original" and add the label
                    features = {key: val for (key, val) in six.iteritems(result) if key.startswith('original')}
                    features['label'] = label

                    # Append to the overall DataFrame
                    features_dataframe = pd.concat([features_dataframe, pd.DataFrame([features])], ignore_index=True)
        return features_dataframe
