import torch
import matplotlib.pyplot as plt

class SegmentationTrainer:
    # This class handles the training and evaluation loop for the segmentation model.
    # It implements:
    #   - A training loop with early stopping based on validation loss.
    #   - Evaluation on a test set.
    #   - Plotting functions for loss curves and performance metrics (IoU and accuracy).

    def __init__(self, model_wrapper, train_loader, val_loader, epochs=100, patience=15):
        """
        Args:
            model_wrapper (SegmentationModelWrapper): The model wrapper object containing the model and its configurations.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            epochs (int): Maximum number of training epochs.
            patience (int): Number of epochs with no improvement before early stopping.
        """
        self.model_wrapper = model_wrapper
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.patience = patience

    def train(self):
        model = self.model_wrapper.model
        criterion = self.model_wrapper.criterion
        optimizer = self.model_wrapper.optimizer
        scheduler = self.model_wrapper.scheduler
        device = self.model_wrapper.device

        train_losses, val_losses = [], []
        val_ious, val_accuracies = [], []
        best_val_loss = float('inf')
        no_improvement_count = 0
        best_model_wts = model.state_dict()

        for epoch in range(self.epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0
            for images, masks, _ in self.train_loader:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item() * images.size(0)
            epoch_train_loss /= len(self.train_loader.dataset)
            train_losses.append(epoch_train_loss)

            # Validation phase
            model.eval()
            epoch_val_loss, epoch_val_iou, epoch_val_accuracy = 0, 0, 0
            with torch.no_grad():
                for images, masks, _ in self.val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)['out']
                    loss = criterion(outputs, masks)
                    epoch_val_loss += loss.item() * images.size(0)

                    preds = (outputs > 0).float()
                    epoch_val_iou += self.model_wrapper.calculate_iou(preds, masks)
                    epoch_val_accuracy += self.model_wrapper.calculate_accuracy(preds, masks)

            epoch_val_loss /= len(self.val_loader.dataset)
            val_losses.append(epoch_val_loss)
            val_ious.append(epoch_val_iou / len(self.val_loader))
            val_accuracies.append(epoch_val_accuracy / len(self.val_loader))

            print(f'Epoch {epoch + 1}/{self.epochs}, Train Loss: {epoch_train_loss:.4f}, '
                  f'Val Loss: {epoch_val_loss:.4f}, Val IoU: {epoch_val_iou / len(self.val_loader):.4f}, '
                  f'Val Accuracy: {epoch_val_accuracy / len(self.val_loader):.4f}')

            # Early stopping check
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                no_improvement_count = 0
                best_model_wts = model.state_dict()
            else:
                no_improvement_count += 1
            if no_improvement_count >= self.patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

            scheduler.step()

        # Load best model weights from training.
        model.load_state_dict(best_model_wts)
        return train_losses, val_losses, val_ious, val_accuracies

    def evaluate(self, test_loader):
        """
        Evaluate the trained model on the test dataset.
        Returns:
            test_loss, test_iou, test_accuracy
        """
        model = self.model_wrapper.model
        criterion = self.model_wrapper.criterion
        device = self.model_wrapper.device
        model.eval()
        test_loss, test_iou, test_accuracy = 0, 0, 0

        with torch.no_grad():
            for images, masks, _ in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                test_loss += loss.item() * images.size(0)
                preds = (outputs > 0).float()
                test_iou += self.model_wrapper.calculate_iou(preds, masks)
                test_accuracy += self.model_wrapper.calculate_accuracy(preds, masks)

        test_loss /= len(test_loader.dataset)
        test_iou /= len(test_loader)
        test_accuracy /= len(test_loader)
        print(f'Test Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}, Test Accuracy: {test_accuracy:.4f}')
        return test_loss, test_iou, test_accuracy

    def plot_loss_curves(self, train_losses, val_losses):
        """Plot the training and validation loss curves."""
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Train and Validation Loss')
        plt.show()

    def plot_metrics_curves(self, val_ious, val_accuracies):
        """Plot the IoU and accuracy curves for validation."""
        plt.figure(figsize=(10, 5))
        plt.plot(val_ious, label='Validation IoU')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.legend()
        plt.title('Validation IoU and Accuracy')
        plt.show()
