import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PTBXLDataset(Dataset):
    def __init__(self, data_path, mode='train'):
        """
        data_path: Directory containing the .npy files
        mode: 'train', 'val', or 'test'
        """
        super().__init__()
        self.mode = mode

        gt_name = f"{mode}_data_gt.npy"
        self.data_file = os.path.join(data_path, gt_name)
        self.data = np.load(self.data_file, mmap_mode='r')
        self.num_samples = self.data.shape[0]

        mask_name = f"{mode}_data_mask.npy"
        mask_file = os.path.join(data_path, mask_name)
        if mode == "train":
            if not os.path.isfile(mask_file):
                raise FileNotFoundError(f"训练需要观测 npy: {mask_file}")
            self.mask_data = np.load(mask_file, mmap_mode="r")
            if self.mask_data.shape[0] != self.num_samples:
                raise ValueError(
                    f"{mask_name} N={self.mask_data.shape[0]} 与 {gt_name} N={self.num_samples} 不一致"
                )
            if self.mask_data.shape[1:] != self.data.shape[1:]:
                raise ValueError(
                    f"{mask_name} 形状 {self.mask_data.shape} 与 {gt_name} {self.data.shape} 不一致"
                )
            self._dual = True
        else:
            self.mask_data = None
            self._dual = False

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Read the sample: shape (5000, 12)
        gt = np.array(self.data[idx], dtype=np.float32)
        gt = torch.from_numpy(gt).transpose(0, 1)  # (12, 5000)

        if self._dual:
            obs = np.array(self.mask_data[idx], dtype=np.float32)
            obs = torch.from_numpy(obs).transpose(0, 1)
            return gt, obs

        return gt
