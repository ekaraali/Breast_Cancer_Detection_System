import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

class DeepLabHead(nn.Sequential):
    # A custom segmentation head to replace the default classifier in DeepLabV3.
    # This head consists of a Conv2D -> BatchNorm -> ReLU sequence followed by a final Conv2D layer.
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )

class SegmentationModelWrapper:
    # This class encapsulates the DeepLabV3 model:
    # - It modifies the first convolutional layer to accept grayscale (single-channel) images.
    # - It replaces the classifier with a custom DeepLabHead.
    # - It sets up the loss function, optimizer, and learning rate scheduler.
    # - It provides helper functions for calculating metrics, saving/loading models, and visualization.

    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.initialize_model().to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def initialize_model(self):
        model = deeplabv3_resnet50(pretrained=True)
        # Modify the first conv layer for single-channel input (grayscale).
        model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Replace the classifier with our custom head to produce a single-channel mask.
        model.classifier = DeepLabHead(2048, 1)
        # Remove the auxiliary classifier if it's not needed.
        model.aux_classifier = None
        return model

    @staticmethod
    def calculate_iou(preds, masks):
        """Calculate the Intersection over Union (IoU) metric."""
        intersection = torch.logical_and(preds, masks).float().sum((1, 2))
        union = torch.logical_or(preds, masks).float().sum((1, 2))
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean().item()

    @staticmethod
    def calculate_accuracy(preds, masks):
        """Calculate the pixel-wise accuracy."""
        preds = (preds > 0.5).float()
        correct = (preds == masks).float().sum()
        accuracy = correct / masks.numel()
        return accuracy.item()

    def apply_threshold(self, mask, threshold=0.5):
        """Apply thresholding to convert output probabilities into binary mask."""
        return (mask > threshold).float()

    def save_model(self, path):
        """Save model weights to the specified path."""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """Load model weights from the specified path."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def visualize_results(self, test_loader, threshold=0.5, num_images=5):
        """
        Visualize the test results including the input image, true mask, and predicted mask.
        """
        self.model.eval()
        images_so_far = 0
        import matplotlib.pyplot as plt  # Local import for plotting
        plt.figure(figsize=(20, num_images * 5))
        with torch.no_grad():
            for images, masks, img_names in test_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self.model(images)['out']
                outputs = self.apply_threshold(outputs, threshold)
                for j in range(images.size(0)):
                    images_so_far += 1
                    ax = plt.subplot(num_images, 3, images_so_far * 3 - 2)
                    ax.imshow(images.cpu()[j].numpy().squeeze(), cmap='gray')
                    ax.set_title(f'Input: {img_names[j]}')
                    ax.axis('off')

                    ax = plt.subplot(num_images, 3, images_so_far * 3 - 1)
                    ax.imshow(masks.cpu()[j].numpy().squeeze(), cmap='gray')
                    ax.set_title('True Mask')
                    ax.axis('off')

                    ax = plt.subplot(num_images, 3, images_so_far * 3)
                    ax.imshow(outputs.cpu()[j].numpy().squeeze(), cmap='gray')
                    ax.set_title('Predicted Mask')
                    ax.axis('off')

                    if images_so_far == num_images:
                        plt.tight_layout()
                        plt.show()
                        return
        plt.tight_layout()
        plt.show()
