import os
import pandas as pd
import argparse
import random
from src.utils import *

'''
Unified csv creation for both training and inference datasets.
'''

LABEL_MAP = {
    'Garbage': 0,
    'Multi': 1,
    'Single': 2
}

def preprocess_dataset(src_root, mode="train", frac_train=0.8, output_file=None):

    src_root = os.path.abspath(src_root)

    # INFERENCE MODE
    if mode == "inference":
        pcd_dir = os.path.join(src_root, "data/Part")

        files = [f for f in os.listdir(pcd_dir) if f.lower().endswith(".pcd")]
        df = pd.DataFrame({
            "data": [os.path.join("data/Part", f) for f in files],
            "label": [0] * len(files)
        })

        if output_file is None:
            output_file = os.path.join(src_root, "modeltrees_inference.csv")

        df.to_csv(output_file, sep=";", index=False)
        print(f"[Inference] Saved file list to {output_file}")
        return df

    # TRAINING MODE
    df_train = pd.DataFrame(columns=["data", "label"])
    df_test  = pd.DataFrame(columns=["data", "label"])

    for folder, label in LABEL_MAP.items():
        class_path = os.path.join(src_root, folder)
        if not os.path.isdir(class_path):
            print(f"Missing expected class folder: {folder}")
            continue

        files = [f for f in os.listdir(class_path) if f.lower().endswith(".pcd")]
        files = [f for f in files if not f.startswith('.')]
        random.shuffle(files)

        n_train = int(len(files) * frac_train)
        train_files = files[:n_train]
        test_files  = files[n_train:]

        df_train = pd.concat([df_train,
            pd.DataFrame({"data": [f"{folder}/{f}" for f in train_files],
                          "label": [label] * len(train_files)})],
            ignore_index=True)

        df_test = pd.concat([df_test,
            pd.DataFrame({"data": [f"{folder}/{f}" for f in test_files],
                          "label": [label] * len(test_files)})],
            ignore_index=True)
    # Save
    train_csv = os.path.join(src_root, "modeltrees_train.csv")
    test_csv = os.path.join(src_root, "modeltrees_test.csv")

    df_train.to_csv(train_csv, sep=";", index=False)
    df_test.to_csv(test_csv, sep=";", index=False)

    print(f"[Training] Wrote: {train_csv}")
    print(f"[Training] Wrote: {test_csv}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='source root folder')
    parser.add_argument('--mode', choices=['train', 'inference'], default='train')
    parser.add_argument('--frac', type=float, default=0.8)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--out', default=None, help='optional output csv path')
    args = parser.parse_args()
    preprocess_dataset(args.src, mode=args.mode, frac_train=args.frac, do_augment=args.augment, out_name=args.out)
