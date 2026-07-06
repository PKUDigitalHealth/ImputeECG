import torch
from torch.utils.data import Dataset
import numpy as np

class PTBXLDataset(Dataset):
    def __init__(self, data_path, label_path):
        """
        PTB-XL Dataset
        :param data_path: path to the .npy data file
        :param label_path: path to the .npy label file
        """
        # Use mmap_mode='r' to avoid loading large files entirely into memory
        self.data = np.load(data_path, mmap_mode='r')
        self.labels = np.load(label_path, mmap_mode='r')
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # Original data shape: (N, 5000, 12)
        # Net1D expects: (N, 12, 5000)
        x = self.data[idx].transpose(1, 0)
        
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return x, y
