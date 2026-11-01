"""
Alzheimer's MRI Dataset Loader and Preprocessing Pipeline
Handles standard 4-class folder structures:
  dataset/
    ├── MildDemented/
    ├── ModerateDemented/
    ├── NonDemented/
    └── VeryMildDemented/
"""

import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented"
]

CLASS_LABELS = {
    "MildDemented": 0,
    "ModerateDemented": 1,
    "NonDemented": 2,
    "VeryMildDemented": 3
}

DISPLAY_NAMES = [
    "Mild Impairment (Mild Demented)",
    "Moderate Impairment (Moderate Demented)",
    "Normal Control (Non-Demented)",
    "Early Impairment (Very Mild Demented)"
]


def get_transforms(img_size=224, is_train=True):
    """
    Data transformations for medical brain MRI scans:
    - Random affine/rotations to simulate slight patient head positioning variations
    - Contrast/brightness jittering to simulate varying scanner magnet strengths
    - Normalization with ImageNet standard or custom brain MRI stats
    """
    if is_train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.RandomResizedCrop(img_size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class AlzheimerMRIDataset(Dataset):
    """PyTorch Dataset for Alzheimer MRI images."""
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def load_dataset_splits(data_dir, img_size=224, batch_size=32, val_size=0.15, test_size=0.15, random_state=42):
    """
    Scans data directory, maps subfolders to classes, and returns train, val, test loaders.
    """
    all_files = []
    all_labels = []

    # Map possible folder name variations
    dir_to_class = {
        "milddemented": 0, "mild": 0, "mild_demented": 0,
        "moderatedemented": 1, "moderate": 1, "moderate_demented": 1,
        "nondemented": 2, "normal": 2, "non_demented": 2, "control": 2,
        "verymilddemented": 3, "verymild": 3, "very_mild_demented": 3
    }

    for root, dirs, files in os.walk(data_dir):
        folder_name = os.path.basename(root).lower().replace(" ", "").replace("-", "")
        if folder_name in dir_to_class:
            c_idx = dir_to_class[folder_name]
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
                    all_files.append(os.path.join(root, f))
                    all_labels.append(c_idx)

    if len(all_files) == 0:
        raise ValueError(f"No MRI images found in {data_dir}. Ensure folders like 'NonDemented', 'MildDemented', etc. exist.")

    # Train / Val / Test split
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, all_labels, test_size=test_size, stratify=all_labels, random_state=random_state
    )

    adjusted_val_size = val_size / (1.0 - test_size)
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels, test_size=adjusted_val_size, stratify=train_labels, random_state=random_state
    )

    train_ds = AlzheimerMRIDataset(train_files, train_labels, transform=get_transforms(img_size, is_train=True))
    val_ds = AlzheimerMRIDataset(val_files, val_labels, transform=get_transforms(img_size, is_train=False))
    test_ds = AlzheimerMRIDataset(test_files, test_labels, transform=get_transforms(img_size, is_train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Dataset summary from {data_dir}:")
    print(f"  Training samples:   {len(train_ds)}")
    print(f"  Validation samples: {len(val_ds)}")
    print(f"  Testing samples:    {len(test_ds)}")

    return train_loader, val_loader, test_loader
