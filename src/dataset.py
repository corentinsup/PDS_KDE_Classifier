import os
import shutil
import numpy as np
import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset
import open3d as o3d
import concurrent.futures
from functools import partial
from tqdm import tqdm
from src.utils import read_pcd_with_fields


class ModelTreesDataLoader(Dataset):
    def __init__(self, csvfile, root_dir, split, transform, do_update_caching, kde_transform, frac=1.0, result_dir='results', verbose=True):
        """
            Arguments:
                :param csv_file (string): Path to the csv file with annotations
                :param root_dir (string): Directory with the csv files and the folders containing pcd files per class
                :param split (string): type of dataset (train or test)
                :param transform (callable, optional): Optional transform to be applied
                :param frac (float, optional): fraction of the data loaded
                    on a sample.
        """
        # create code for caching grids
        self.transform = transform
        self.root_dir = root_dir
        pickle_dir = root_dir + 'tmp_grids_' + split + "/"
        self.pickle_dir = pickle_dir
        if do_update_caching:
            self.clean_temp()
            os.mkdir(pickle_dir)
            if split != 'inference':
                os.mkdir(pickle_dir + "Garbage")
                os.mkdir(pickle_dir + "Multi")
                os.mkdir(pickle_dir + "Single")
            else:
                os.mkdir(pickle_dir + "data")
        self.data = pd.read_csv(root_dir + csvfile, delimiter=';')

        if verbose:
            print('Loading ', split, ' set...')
        self.num_fails = []
        if do_update_caching:
            # creating grids using multiprocess
            with concurrent.futures.ProcessPoolExecutor() as executor:
                partialmapToKDE = partial(self.mapToKDE, root_dir, pickle_dir, kde_transform)
                args = range(len(self.data))
                results = list(tqdm(executor.map(partialmapToKDE, args), total=len(self.data), smoothing=.9, desc="Creating caching files", disable=not verbose))
            self.num_fails = [(idx, x) for (idx, x) in enumerate(results) if x != ""]
            if verbose:
                print(f"Number of failing files: {len(self.num_fails)}")

            # Update self.data and csv files for data and failed_data
            df_failed_data = self.data.iloc[[x for x,_ in self.num_fails]]
            self.data.drop(labels=[x for x,_ in self.num_fails], axis=0, inplace=True)

            # creation of results directory if not existing
            if not os.path.exists(os.path.join(root_dir, result_dir)):
                os.mkdir(os.path.join(root_dir, result_dir))

            # save failed data and updated data csv files
            df_failed_data.to_csv(os.path.join(root_dir, result_dir, "failed_data.csv"), sep=';', index=True, index_label="Index")
            self.data.to_csv(os.path.join(root_dir, csvfile), sep=';', index=False)

        # shuffle the dataset
        self.data = self.data.sample(frac=frac, random_state=42).reset_index(drop=True)
        lst_file_names = [os.path.basename(x) + '.pickle' for x in self.data.data.values]
      
        self.data.data = lst_file_names
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.data.iloc[idx, 0]

        with open(self.pickle_dir + filename, 'rb') as file:
            sample = pickle.load(file)

        sample['label'] = sample.get('label', self.data.iloc[idx, 1])

        if self.transform:
            sample = self.transform(sample)

        return sample

    def clean_temp(self):
        if os.path.exists(self.pickle_dir):
            shutil.rmtree(self.pickle_dir)

    def mapToKDE(self, root_dir, pickle_dir, kde_transform, idx):
        pcd_name = ""
        try:
            samp = self.data.iloc[idx]
            pcd_name = os.path.join(root_dir, samp['data'])

            # read point cloud with all the fields
            data, fields = read_pcd_with_fields(pcd_name)
            idx_inCluster = fields.index('inCluster')
            idx_x, idx_y, idx_z = fields.index('x'), fields.index('y'), fields.index('z')
            xyz_indices = [idx_x, idx_y, idx_z]
            
            # separate cluster points
            cluster_points = data[data[:, idx_inCluster] == 1][:, xyz_indices]
            all_points = data[:, xyz_indices]
    
            label = np.asarray(samp['label'])
            sample = {
            'data_cluster': cluster_points,
            'data_all': all_points,
            'label': label
            }
            
            # apply KDE transform
            sample = kde_transform(sample)

            with open(os.path.join(pickle_dir, os.path.basename(samp['data']) + '.pickle'), 'wb') as file:
                pickle.dump(sample, file)
            return ""
        except Exception as e:
            print(f"Failed to process {pcd_name if pcd_name else idx}: {e}")
            return pcd_name if pcd_name else str(idx)      


def main():
    print("not the right way to use me Pal")


if __name__ == '__main__':
    main()
