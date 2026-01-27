import torch 
import random
import numpy as np
from src.utils import read_pcd_with_fields, ToKDE, RandRotate, RandScale

class TreeCropTransform:
    def __init__(self, crop_ratio=None, probability=None):
        self.crop_ratio = crop_ratio
        self.probability = probability

    def __call__(self, sample):
        #cluster_points = sample["data_cluster"]                # only cluster points
        all_points = sample["data_all"]                         # all points
        cluster_mask = sample["cluster_mask"]
        cluster_points = all_points[cluster_mask]     # get cluster points using mask

        # add a random chance to apply cropping or not but 
        if self.probability is not None:
            if random.random() > self.probability or sample.get("label") == 0:
                # skip cropping
                return sample

        # Compute cluster height from cluster points only
        z_values = cluster_points[:, 2]
        z_min, z_max = z_values.min(), z_values.max()
        cluster_height = z_max - z_min

        # Compute cutoff from ratio
        z_cut = z_min + self.crop_ratio * cluster_height

        # Apply threshold to cluster points only
        height_mask = all_points[:, 2] <= z_cut
        filtered_points = all_points[height_mask]

        # Update sample
        sample["data_all"] = filtered_points
        #sample["data_cluster"] = cluster_points[cluster_points[:, 2] <= z_cut]  # Filtered cluster points
        sample["cluster_mask"] = cluster_mask[height_mask]  # Filter mask to match filtered points
        sample["label"] = 3  # New label for cropped trees
    
        return sample


class TreeRadiusCrop:
    def __init__(self, radius, probability=None):
        self.radius = radius
        self.probability = probability

    def __call__(self, sample):
        """
        Randomly choose a point in the pointCloud, and assign its value to 
        all points within the specified radius.

        Args:
            points (np.ndarray): Nx3 array of point coordinates.
            radius (float): Radius for cropping.

        Returns:
            np.ndarray: Cropped points within the specified radius.
        """
        if self.probability is not None:
            if random.random() > self.probability or sample.get("label") == 0:
                # skip cropping
                return sample
        
        # choose a random point as center
        all_points = sample["data_all"]
        cluster_mask = sample["cluster_mask"]
        cluster_points = all_points[cluster_mask]  # get cluster points using mask

        center_idx = random.randint(0, len(cluster_points) - 1)
        center_point = cluster_points[center_idx]
     
        # compute distances from center point
        distances = np.linalg.norm(all_points - center_point, axis=1)
        inside_mask = distances <= self.radius

        # Filter points and update cluster membership
        sample["data_all"] = all_points[inside_mask]
        sample["cluster_mask"] = cluster_mask[inside_mask]  # Filter mask to match filtered points
        sample["label"] = 3  # New label for radius-cropped trees

        return sample
        
class RandomDropout:
    def __init__(self, drop_probability):
        self.drop_probability = drop_probability

    def __call__(self, sample):
        # get all points
        all_points = sample["data_all"]
        #cluster_points = sample["data_cluster"]
        #cluster_idx = np.array([np.where((all_points == pt).all(axis=1))[0][0] for pt in cluster_points])
        cluster_mask = sample["cluster_mask"]

        drop_ratio = random.uniform(0, self.drop_probability)
        # apply dropout 
        mask = np.random.rand(all_points.shape[0]) > drop_ratio

        if mask.sum() < 10:
            return sample  # avoid too much dropout
        
        sample["data_all"] = all_points[mask]
        #sample["data_cluster"] = all_points[cluster_idx][mask[cluster_idx]]
        #sample["cluster_mask"] = cluster_mask & mask
        sample["cluster_mask"] = cluster_mask[mask]

        return sample

class JitterPoints:
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, sample):
        all_points = sample["data_all"]
        cluster_mask = sample["cluster_mask"]
        cluster_points = all_points[cluster_mask]

        jitter = np.clip(self.sigma * np.random.randn(*all_points.shape), -self.clip, self.clip)
        sample["data_all"] = all_points + jitter
        return sample

class PointCloudTransforms:
    def __init__(self, config):
        self.p = config.training.probability
        self.radius = TreeRadiusCrop(
            radius=config.training.crop_radius,
            probability=self.p
        )
        self.crop = TreeCropTransform(
            crop_ratio=config.training.crop_ratio,
            probability=self.p
        )
        self.kde = ToKDE(
            grid_size=config.shared.grid_size,
            kernel_size=config.shared.kernel_size,
            num_repeat=config.shared.num_repeat_kernel
        )
        self.transforms = [
            JitterPoints(sigma=0.01, clip=0.05),
            RandomDropout(drop_probability=0.1),
        ]

    def __call__(self, sample):
        # apply random crop or radius crop
        if random.random() < self.p:
            sample = self.radius(sample)
        else:
            sample = self.crop(sample)

        # apply other point transforms
        for transform in self.transforms:
            #print("Applying transform:", transform.__class__.__name__)
            sample = transform(sample)
        
        # apply KDE voxelization
        sample = self.kde(sample)

        return sample

class VoxelTransforms:
    def __init__(self, config):
        self.transforms = [
            RandRotate(),
            #RandScale(kernel_size=config.shared.kernel_size),
        ]

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample
    
