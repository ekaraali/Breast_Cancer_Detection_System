import os
from PIL import Image
from torch.utils.data import Dataset

class BreastCancerDataset(Dataset):
    # A custom Dataset class to load breast cancer image data along with their corresponding masks.
    # It supports optional augmentation (using Albumentations) and applies specified transforms.

    def __init__(self, image_dir, transform=None, augment=False):
        """
        Args:
            image_dir (str): Path to the folder containing images.
            transform (callable, optional): A function/transform to apply to the images and masks.
            augment (bool): If True, perform augmentation using an Albumentations pipeline.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.augment = augment
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') and not f.endswith('_mask.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.png', '_mask.png')
        mask_path = os.path.join(self.image_dir, mask_name)

        # Open image and mask in grayscale mode.
        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        # Apply augmentation if specified.
        if self.augment:
            augmented = self.augment(image=np.array(image), mask=np.array(mask))
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])

        # Apply transformations if provided.
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Convert mask to binary: values 0 or 1.
        mask = (mask > 0.5).float()
        return image, mask, img_name
