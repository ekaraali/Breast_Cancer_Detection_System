from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from model import SegmentationModelWrapper
from trainer import SegmentationTrainer

class CrossValidator:
    # This class implements K-Fold cross validation:
    #   - It splits the combined dataset into training and validation subsets.
    #   - It creates a new model instance for each fold and trains it using SegmentationTrainer.
    #   - It stores and returns the training metrics for each fold.
    def __init__(self, combined_dataset, num_folds=5, batch_size=16):
        """
        Args:
            combined_dataset (Dataset): The combined dataset containing all images.
            num_folds (int): Number of cross-validation folds.
            batch_size (int): Batch size for DataLoaders.
        """
        self.combined_dataset = combined_dataset
        self.num_folds = num_folds
        self.batch_size = batch_size
        self.kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        self.fold_results = []

    def run(self, epochs=100, patience=15):
        fold = 0
        for train_idx, val_idx in self.kf.split(self.combined_dataset):
            fold += 1
            print(f'\nFold {fold}/{self.num_folds}')
            train_subset = Subset(self.combined_dataset, train_idx)
            val_subset = Subset(self.combined_dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)

            # Create a new model instance for the current fold.
            model_wrapper = SegmentationModelWrapper()
            trainer = SegmentationTrainer(model_wrapper, train_loader, val_loader, epochs, patience)
            train_losses, val_losses, val_ious, val_accuracies = trainer.train()

            self.fold_results.append({
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_ious': val_ious,
                'val_accuracies': val_accuracies
            })

            trainer.plot_loss_curves(train_losses, val_losses)
            trainer.plot_metrics_curves(val_ious, val_accuracies)
        return self.fold_results
