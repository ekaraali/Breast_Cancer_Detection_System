import os
import pandas as pd
from radiomics_extractor import RadiomicsFeatureExtractor
from preprocessor import FeaturePreprocessor
from svm_classifier import SVMClassifier, ThresholdAdjustedModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle


def main():
    # Define directories for benign and malignant masks (adjust paths accordingly)
    benign_mask_dir = '/path/to/benign'
    malignant_mask_dir = '/path/to/malignant'

    # Initialize the radiomics feature extractor
    radiomics_extractor = RadiomicsFeatureExtractor()

    # Extract features from both directories and combine them into one DataFrame
    benign_features_df = radiomics_extractor.extract_features_from_folder(benign_mask_dir, label=0)
    malignant_features_df = radiomics_extractor.extract_features_from_folder(malignant_mask_dir, label=1)
    features_dataframe = pd.concat([benign_features_df, malignant_features_df], ignore_index=True)
    print("Features extracted and stored in DataFrame.")

    # Optionally, save the features to CSV
    # features_dataframe.to_csv("radiomics_features.csv", index=False)

    # Separate labels and features
    labels = features_dataframe['label']
    features = features_dataframe.drop(columns=['label'])

    # Preprocess features: normalization and outlier filtering
    preprocessor = FeaturePreprocessor(z_score_threshold=2.5, outlier_fraction_threshold=0.1)
    features_normalized = preprocessor.normalize_features(features)
    filtered_features = preprocessor.filter_outliers(features_normalized)
    filtered_labels = labels[filtered_features.index]

    # Split into training and test sets BEFORE applying SMOTE
    X_train, X_test, y_train, y_test = train_test_split(filtered_features, filtered_labels, test_size=0.2,
                                                        random_state=42)

    # Now apply SMOTE only on the training set
    X_train, y_train = preprocessor.apply_smote(X_train, y_train)

    # Perform feature selection: choose top k features
    k = 77
    # Ensure X_train is a DataFrame (to retain column names)
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=filtered_features.columns)
        X_test = pd.DataFrame(X_test, columns=filtered_features.columns)
    X_train_selected, X_test_selected, selected_feature_names = preprocessor.select_features(X_train, y_train, X_test,
                                                                                             k)
    print("Selected feature names:")
    print(selected_feature_names)

    # Initialize and train the SVM classifier using cross-validation
    svm_classifier = SVMClassifier()
    svm_classifier.cross_validate(X_train_selected, y_train)

    # Evaluate the best model on the test set
    svm_classifier.evaluate(X_test_selected, y_test)

    # Save the best model to file
    model_filename = 'svm_model_best_1.pkl'
    svm_classifier.save_model(model_filename)

    # Create a threshold-adjusted model instance (adjust threshold as needed)
    best_threshold_model = ThresholdAdjustedModel(svm_classifier.best_model, threshold=0.5554)

    # Predict on the test set using the threshold-adjusted model and print a classification report
    y_test_pred_adj = best_threshold_model.predict(X_test_selected)
    print("Threshold Adjusted Model Evaluation:")
    print(classification_report(y_test, y_test_pred_adj))

    # Save the threshold-adjusted model to file
    model_filename_adjusted = 'svm_model_best_last.pkl'
    with open(model_filename_adjusted, 'wb') as file:
        pickle.dump(best_threshold_model, file)


if __name__ == "__main__":
    main()
