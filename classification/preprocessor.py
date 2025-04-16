import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, mutual_info_classif

class FeaturePreprocessor:
    """
    FeaturePreprocessor handles normalization, outlier filtering, oversampling, and feature selection.
    """

    def __init__(self, z_score_threshold=2.5, outlier_fraction_threshold=0.1):
        """
        Args:
            z_score_threshold (float): Z-score threshold for detecting outliers.
            outlier_fraction_threshold (float): Maximum allowed fraction of features exceeding the threshold.
        """
        self.z_score_threshold = z_score_threshold
        self.outlier_fraction_threshold = outlier_fraction_threshold

    def normalize_features(self, features):
        """
        Normalize features using z-score normalization.
        """
        return (features - features.mean()) / features.std()

    def filter_outliers(self, features_normalized):
        """
        Filter out rows where the fraction of features exceeding the z-score threshold is too high.
        """
        exceeds_threshold = np.abs(features_normalized) > self.z_score_threshold
        num_exceeding_per_row = np.sum(exceeds_threshold, axis=1)
        total_features = features_normalized.shape[1]
        rows_to_keep = (num_exceeding_per_row / total_features) < self.outlier_fraction_threshold
        return features_normalized[rows_to_keep]

    def apply_smote(self, features, labels, random_state=42):
        """
        Apply SMOTE to balance the dataset by increasing samples of the minority class.
        """
        smote = SMOTE(sampling_strategy={1: sum(labels == 0)}, random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(features, labels)
        return X_resampled, y_resampled

    def select_features(self, X_train, y_train, X_test, k):
        """
        Select the top k features using mutual information.

        Returns:
            X_train_selected, X_test_selected, selected_feature_names
        """
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        selected_features_mask = selector.get_support()
        selected_feature_names = X_train.columns[selected_features_mask] if isinstance(X_train, pd.DataFrame) else None
        return X_train_selected, X_test_selected, selected_feature_names
