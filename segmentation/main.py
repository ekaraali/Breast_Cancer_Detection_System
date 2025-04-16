import os
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
import albumentations as A

from dataset import BreastCancerDataset
from cross_validation import CrossValidator
from model import SegmentationModelWrapper
from trainer import SegmentationTrainer

def main():
    # Parameters
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 0.01
    NUM_FOLDS = 5

    # Define image transformations for training, validation, and test sets.
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        'test': transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    }

    # Define augmentation with Albumentations.
    augmentation = A.Compose([
        A.HorizontalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5)
    ], additional_targets={'mask': 'mask'})

    # Directories for your datasets (adjust paths accordingly).
    benign_image_dir = '/path/to/benign'
    malignant_image_dir = '/path/to/malignant'

    # Create individual datasets.
    benign_dataset = BreastCancerDataset(benign_image_dir, transform=data_transforms['train'])
    malignant_dataset = BreastCancerDataset(malignant_image_dir, transform=data_transforms['train'])
    augmented_malignant_dataset = BreastCancerDataset(malignant_image_dir, transform=data_transforms['train'], augment=augmentation)

    # Combine datasets into one.
    combined_dataset = ConcatDataset([benign_dataset, malignant_dataset, augmented_malignant_dataset])

    # Run K-Fold Cross Validation.
    cross_validator = CrossValidator(combined_dataset, num_folds=NUM_FOLDS, batch_size=BATCH_SIZE)
    fold_results = cross_validator.run(epochs=EPOCHS, patience=15)

    # Print summary of results for each fold.
    for fold, result in enumerate(fold_results):
        print(f'\nFold {fold+1} Results:')
        print(f'  Best Validation Loss: {min(result["val_losses"]):.4f}')
        print(f'  Best Validation IoU: {max(result["val_ious"]):.4f}')
        print(f'  Best Validation Accuracy: {max(result["val_accuracies"]):.4f}')

    # Example: Evaluate on a test set using a random split (for demonstration).
    test_size = int(0.2 * len(combined_dataset))
    test_subset, _ = random_split(combined_dataset, [test_size, len(combined_dataset) - test_size])
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

    print('\nEvaluating the model on the test set...')
    best_model_wrapper = SegmentationModelWrapper(learning_rate=LEARNING_RATE)
    # Load your best saved model weights; adjust the path as needed.
    best_model_wrapper.load_model('best_model_1.pth')
    test_trainer = SegmentationTrainer(best_model_wrapper, None, None)
    test_loss, test_iou, test_accuracy = test_trainer.evaluate(test_loader)
    print(f'Test Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # Optionally, visualize the test results.
    best_model_wrapper.visualize_results(test_loader, threshold=0.5, num_images=5)

if __name__ == "__main__":
    main()
