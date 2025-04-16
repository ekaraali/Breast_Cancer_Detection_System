import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

class SVMClassifier:
    """
    SVMClassifier trains an SVM model using GridSearchCV and K-Fold cross-validation.
    It computes performance metrics and stores the best model based on sensitivity.
    """
    def __init__(self, param_grid=None, cv_folds=5, grid_cv_folds=5, random_state=100):
        """
        Args:
            param_grid (dict): Hyperparameter grid for SVC.
            cv_folds (int): Number of folds for outer cross-validation.
            grid_cv_folds (int): Number of folds for GridSearchCV.
            random_state (int): Random state for reproducibility.
        """
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
            }
        self.param_grid = param_grid
        self.cv_folds = cv_folds
        self.grid_cv_folds = grid_cv_folds
        self.random_state = random_state
        self.svm_classifier = SVC(probability=True)
        self.best_sensitivity = 0
        self.best_model = None
        self.train_metrics = {}
        self.val_metrics = {}

    def cross_validate(self, X_train, y_train):
        """
        Perform K-Fold cross-validation with GridSearchCV on the training set.
        The best model is updated based on the highest sensitivity.
        """
        folds = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        train_accuracies = []
        train_sensitivities = []
        train_specificities = []
        val_accuracies = []
        val_sensitivities = []
        val_specificities = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds.split(X_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = np.array(y_train)[train_idx], np.array(y_train)[val_idx]

            grid_search = GridSearchCV(estimator=self.svm_classifier, param_grid=self.param_grid, cv=self.grid_cv_folds)
            grid_search.fit(X_fold_train, y_fold_train)
            best_model_fold = grid_search.best_estimator_
            print("Fold {}: Best Hyperparameters: {}".format(fold_idx+1, grid_search.best_params_))

            # Evaluate on validation set
            val_accuracy = best_model_fold.score(X_fold_val, y_fold_val)
            val_accuracies.append(val_accuracy)
            y_val_pred = best_model_fold.predict(X_fold_val)
            cm_val = confusion_matrix(y_fold_val, y_val_pred)
            if cm_val.shape == (2, 2):
                val_sensitivity = cm_val[1, 1] / (cm_val[1, 1] + cm_val[1, 0])
                val_specificity = cm_val[0, 0] / (cm_val[0, 0] + cm_val[0, 1])
            else:
                val_sensitivity, val_specificity = 0, 0
            val_sensitivities.append(val_sensitivity)
            val_specificities.append(val_specificity)

            # Update best model based on sensitivity
            if val_sensitivity > self.best_sensitivity:
                self.best_sensitivity = val_sensitivity
                self.best_model = best_model_fold

            # Evaluate on training set
            train_accuracy = best_model_fold.score(X_fold_train, y_fold_train)
            train_accuracies.append(train_accuracy)
            y_train_pred = best_model_fold.predict(X_fold_train)
            cm_train = confusion_matrix(y_fold_train, y_train_pred)
            if cm_train.shape == (2, 2):
                train_sensitivity = cm_train[1, 1] / (cm_train[1, 1] + cm_train[1, 0])
                train_specificity = cm_train[0, 0] / (cm_train[0, 0] + cm_train[0, 1])
            else:
                train_sensitivity, train_specificity = 0, 0
            train_sensitivities.append(train_sensitivity)
            train_specificities.append(train_specificity)

        self.train_metrics = {
            'accuracy': np.mean(train_accuracies),
            'sensitivity': np.mean(train_sensitivities),
            'specificity': np.mean(train_specificities)
        }
        self.val_metrics = {
            'accuracy': np.mean(val_accuracies),
            'sensitivity': np.mean(val_sensitivities),
            'specificity': np.mean(val_specificities)
        }
        print("Average Training Metrics:", self.train_metrics)
        print("Average Validation Metrics:", self.val_metrics)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the best model on the test set and print performance metrics.
        """
        if self.best_model is None:
            raise ValueError("No best model found. Run cross_validate() first.")
        y_pred = self.best_model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        cm_test = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", cm_test)
        print("Test Accuracy:", accuracy_score(y_test, y_pred))
        # Plot confusion matrix
        sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def save_model(self, filename):
        """
        Save the best model to a file.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self.best_model, file)

    def load_model(self, filename):
        """
        Load a model from a file.
        """
        with open(filename, 'rb') as file:
            self.best_model = pickle.load(file)
        return self.best_model

class ThresholdAdjustedModel:
    """
    ThresholdAdjustedModel wraps an SVM model to adjust its decision threshold.
    """
    def __init__(self, model, threshold):
        """
        Args:
            model: A trained SVM model.
            threshold (float): Decision threshold to classify samples.
        """
        self.model = model
        self.threshold = threshold

    def predict(self, X):
        """
        Predict class labels using a custom threshold on the decision function.
        """
        decision_scores = self.model.decision_function(X)
        return (decision_scores >= self.threshold).astype(int)

    def predict_proba(self, X):
        """
        Predict probabilities by delegating to the model's predict_proba.
        """
        return self.model.predict_proba(X)
