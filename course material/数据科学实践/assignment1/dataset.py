import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm


class LeavesDataset(Dataset):
    """Custom Leaves Dataset class"""
    
    def __init__(self, root, mode='train', train_ratio=0.7, 
                 val_ratio=0.1, test_ratio=0.2, transforms=None):
        """
        Args:
            root (str): Image directory path
            mode (str): 'train', 'valid', or 'test'
            train_ratio (float): Training set ratio
            val_ratio (float): Validation set ratio
            test_ratio (float): Test set ratio
        """
        self.root = root
        self.mode = mode
        self.transforms = transforms
        
        # Read CSV file
        self.data_df = pd.read_csv(os.path.join(self.root, 'train.csv'))
        self.total_len = len(self.data_df)
        
        # Calculate split sizes
        self.train_len = int(self.total_len * train_ratio)
        self.val_len = int(self.total_len * val_ratio)
        self.test_len = self.total_len - self.train_len - self.val_len
        
        # Split data based on mode
        if mode == 'train':
            self.images = self.data_df['image'][:self.train_len].values
            self.labels = self.data_df['label'][:self.train_len].values
        elif mode == 'valid':
            self.images = self.data_df['image'][self.train_len:self.train_len + self.val_len].values
            self.labels = self.data_df['label'][self.train_len:self.train_len + self.val_len].values
        elif mode == 'test':
            self.images = self.data_df['image'][self.train_len + self.val_len:].values
            self.labels = self.data_df['label'][self.train_len + self.val_len:].values
        else:
            raise ValueError('you need to specify a model')
            
        # Create label mapping
        self.unique_labels = sorted(set(self.data_df['label']))
        self.class_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.idx_to_class = {idx: label for label, idx in self.class_to_idx.items()}
            
        print(f'Finished loading {mode} set: {len(self.images)} samples')

    def __getitem__(self, index):
        """Get a single sample"""
        img_path = os.path.join(self.root, self.images[index])
        
        # Read and convert image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None
            
        # Apply transforms
        if self.transforms is not None:
            image =  self.transforms(image)

        label = self.class_to_idx[self.labels[index]]
        return image, label

    def __len__(self):
        """Return dataset size"""
        return len(self.images)


if __name__=='__main__':
    # Data paths and parameters
    root = 'leafdataset'
    batch_size = 128

    # Define transforms outside the class
    transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = LeavesDataset(root, mode='train', transforms=transforms_train)
    val_dataset = LeavesDataset(root, mode='valid', transforms=transforms_test)
    test_dataset = LeavesDataset(root, mode='test', transforms=transforms_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Check dataset
    print(f"Train batches: {len(train_loader)}")
    print(f"Valid batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Number of classes: {len(train_dataset.class_to_idx)}")