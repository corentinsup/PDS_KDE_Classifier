import os
import shutil
import numpy as np
import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from src.utils import read_pcd_with_fields

def mapToKDE(args):
    root_dir, pickle_dir, kde_transform, row = args
    pcd_name = os.path.join(root_dir, row["data"])

    try:
        #print(f"[Worker PID {os.getpid()}] Processing: {pcd_name} → {save_path}")
        # Load .pcd file
        data, fields = read_pcd_with_fields(pcd_name)
        idx_inCluster = fields.index("inCluster")
        xyz_indices = [fields.index("x"), fields.index("y"), fields.index("z")]

        # Separate cluster points
        cluster_points = data[data[:, idx_inCluster] == 1][:, xyz_indices]
        all_points = data[:, xyz_indices]

        label = np.asarray(row["label"])
        sample = {
            "data_cluster": cluster_points,
            "data_all": all_points,
            "label": label,
        }

        # Apply KDE transform
        sample = kde_transform(sample)

        # Save pickle
        save_path = os.path.join(pickle_dir, os.path.basename(row["data"]) + ".pickle")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(sample, f)

        return ""
        print(f"[Worker PID {os.getpid()}] Finished: {pcd_name}")
    except Exception as e:
        return f"{pcd_name}: {e}"

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
        
        self.transform = transform
        self.root_dir = os.path.abspath(root_dir)
        self.split = split
        self.kde_transform = kde_transform

        # create code for caching grids
        self.pickle_dir = os.path.join(self.root_dir, 'tmp_grids_' + split + "/")
        '''if do_update_caching:
            self.clean_temp()
            os.makedirs(self.pickle_dir, exist_ok=True)

            if split != "inference":
                for sub in ['Garbage','Multi', 'Single']:
                    os.makedirs(os.path.join(self.pickle_dir, sub), exist_ok=True)
            else:
                os.makedirs(self.pickle_dir , exist_ok=True)'''

        # Load csv file
        self.data = pd.read_csv(os.path.join(root_dir, csvfile), delimiter=';')

        if verbose:
            print('Loading ', split, ' set...')
        
        # cache building
        self.num_fails = []
        if do_update_caching:
            self.clean_temp()
            os.makedirs(self.pickle_dir, exist_ok=True)
            args = [
                (self.root_dir, self.pickle_dir, self.kde_transform, row)
                for _, row in self.data.iterrows()
            ]
            multiprocessing.set_start_method('spawn', force=True)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(
                    tqdm(
                        executor.map(mapToKDE, args),
                        total=len(args),
                        smoothing=0.9,
                        desc="Creating caching files",
                        disable=not verbose,
                    )
                )
            # Collect failed files
            self.num_fails = [(idx, x) for (idx, x) in enumerate(results) if x != ""]
            if verbose:
                print(f"Number of failing files: {len(self.num_fails)}")

            # Save failed data
            if self.num_fails:
                failed_indices = [x for x, _ in self.num_fails]
                df_failed = self.data.iloc[failed_indices]
                self.data.drop(index=failed_indices, inplace=True)

                os.makedirs(os.path.join(self.root_dir, result_dir), exist_ok=True)
                df_failed.to_csv(
                    os.path.join(self.root_dir, result_dir, "failed_data.csv"),
                    sep=";",
                    index=True,
                    index_label="Index",
                )
                # Save updated data CSV
                self.data.to_csv(os.path.join(root_dir, csvfile), sep=";", index=False)

        # shuffle the dataset
        self.data = self.data.sample(frac=frac, random_state=42).reset_index(drop=True)
        self.data["data"] = [os.path.basename(str(x)) + ".pickle" for x in self.data["data"].values]
      
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.data.iloc[idx]["data"]
        label = self.data.iloc[idx]["label"]

        pickle_path = os.path.join(self.pickle_dir, filename)
        with open(pickle_path, "rb") as file:
            sample = pickle.load(file)

        sample["label"] = sample.get("label", label)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def clean_temp(self):
        if os.path.exists(self.pickle_dir):
            shutil.rmtree(self.pickle_dir)

def main():
    print("not the right way to use me Pal")


if __name__ == '__main__':
    main()
