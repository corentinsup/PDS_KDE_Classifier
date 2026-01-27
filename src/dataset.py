import os
import shutil
import random
import numpy as np
import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from src.pcd_to_pickle import read_pcd_with_fields

class TrainPCDDataset(Dataset):
    def __init__(self, csvfile, root_dir, n_augmentations=1, points_transforms=None, voxel_transforms=None):
        """
            Dataset for only training PCD files with on-the-fly KDE voxelization.
            n_augmentations: number of augmented versions per original sample (default=1 for no augmentation)
        """

        self.root_dir = root_dir
        self.points_transforms = points_transforms
        self.voxel_transforms = voxel_transforms
        self.n_augmentations = n_augmentations

        data = pd.read_csv(csvfile, delimiter=';')
        self.filenames = data["data"].astype(str).values
        self.labels = data["label"].astype(int).values

    def __len__(self): 
        return len(self.filenames) * self.n_augmentations

    def __getitem__(self, idx):
        # Map augmented index back to original sample
        original_idx = idx // self.n_augmentations

        pcd_path = os.path.join(self.root_dir, self.filenames[original_idx])
        data, fields = read_pcd_with_fields(pcd_path)

        idx_inCluster = fields.index("inCluster")
        xyz_indices = [fields.index("x"), fields.index("y"), fields.index("z")]
        in_cluster_mask = data[:, idx_inCluster].astype(bool)
        all_points = data[:, xyz_indices]
        
        sample = {
            "cluster_mask": in_cluster_mask,
            "data_all": all_points,
            "label": self.labels[original_idx],
        }
        # apply crop transformation on point cloud + voxelization
        if self.points_transforms:
                sample = self.points_transforms(sample)
    
        # apply after voxelization transformation
        if self.voxel_transforms:
            sample = self.voxel_transforms(sample)
            
        return sample

class InferenceDataset(Dataset):
    def __init__(self, csvfile, pickle_dir):
        """
            Dataset for inference using precomputed pickles.
        """
        self.pickle_dir = os.path.abspath(pickle_dir)

        # Load csv file
        print(f"Reading CSV file from {csvfile}...")
        data = pd.read_csv(csvfile, delimiter=';')

        # store arrays (avoid large pandas objects per worker)
        self.filenames = data["data"].astype(str).values
        self.labels = data["label"].astype(int).values
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = os.path.join(self.pickle_dir, self.filenames[idx])
        label = self.labels[idx]

        with open(path, "rb") as f:
            sample = pickle.load(f)

        sample["label"] = sample.get("label", label)

        return sample
