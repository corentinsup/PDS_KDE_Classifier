import os
import csv
import pandas as pd
import torchvision.transforms as T
import torch
import time

from packaging import version
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils import *
from models.model_og import KDE_cls_model
from src.csv_creation import preprocess_dataset
from src.dataset_train_inference import InferenceDataset
from src.pcd_to_pickle import csv_to_pickles
from config.config import *

def inference(config) :
    ''' Run inference on the dataset specified in the config file. 
    Args:
        config: configuration object containing parameters for inference.
    Returns:
        None
    '''
    # check torch and if cuda is available
    print("torch version : " + torch.__version__)
    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE")
        device = torch.device('cpu')
    else:
        print("Cuda available")
        device = torch.device('cuda')

    # data transformations
    kde_transform = ToKDE(
        config.shared.grid_size, 
        config.shared.kernel_size, 
        config.shared.num_repeat_kernel
        )
    # Optional: preprocessing of data
    if config.inference.do_preprocess:
        print("Preprocessing dataset...")
        preprocess_dataset(
            src_root=config.inference.src_inf_root,
            mode='inference',
            frac_train=1.0,
            output_file=None
        )
    # Optional: creation of pickles from pcd files
    if config.inference.do_update_caching:
        print("Converting PCD to Pickle for inference set...")
        csv_to_pickles(
            csvfile=os.path.join(config.inference.src_inf_root, config.inference.inference_file),
            root_dir=config.inference.src_inf_root,
            split='inference',
            kde_transform=kde_transform,
            pickle_subdir='pickles_inference',
            verbose=True
        )
    
    # mapping from label index to label name
    shape_names_path = os.path.join(config.inference.src_inf_root, 'modeltrees_shape_names.txt')

    with open(shape_names_path, 'r') as f:
        sample_labels = f.read().splitlines()

    # store relation between number and class label
    dict_labels = {idx: cls for idx, cls in enumerate(sample_labels)}
    print("Class labels mapping:", dict_labels)

    # load model for inference
    if config.inference.verbose:
        print("Loading model for inference...")
    
    conf = {
        "num_class": config.shared.num_class,
        "grid_dim": config.shared.grid_size,
    }
    model = KDE_cls_model(conf).to(device)

    if version.parse(torch.__version__) >= version.parse("2.1.0"):
        checkpoint = torch.load(config.inference.src_model, weights_only=False)
    else:
        checkpoint = torch.load(config.inference.src_model)

    pretrained = checkpoint["model_state_dict"]
    model_dict = model.state_dict()
    
    # add randomly initialized weigths
    # layers to be updated with a fourth class 
    layers = [
        "conv13.weight",
        "bn13.weight",
        "bn13.bias",
        "bn13.running_mean",
        "bn13.running_var",
        "fc4.weight"
    ]
    seed = torch.manual_seed(42)
    for name, param in list(pretrained.items()):
        if name == "conv13.weight":
            old = param
            C, A, K1, K2, K3 = old.shape
            new_filter = torch.randn(1, A, K1, K2, K3, device=device) * 0.01
            pretrained[name] = torch.cat([old, new_filter], dim=0)

        elif name in [
            "bn13.weight", "bn13.bias",
            "bn13.running_mean", "bn13.running_var"
        ]:
            old = param
            new_value = torch.zeros(1, device=device)
            pretrained[name] = torch.cat([old, new_value], dim=0)

        elif name == "fc4.weight":
            old = param
            C, D = old.shape
            new_row = torch.randn(1, D, device=device) * 0.01
            pretrained[name] = torch.cat([old, new_row], dim=0)

    # Update other weights
    model_dict.update(pretrained)
    model.load_state_dict(model_dict, strict=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # create dataLoader for inference set
    print("Creating dataloader for inference set...")
    inference_dataset = InferenceDataset(
        csvfile=os.path.join(config.inference.src_inf_root, "modeltrees_inference_pickles.csv"),
        pickle_dir=os.path.join(config.inference.src_inf_root, 'pickles_inference')
    )

    inferenceDataLoader = DataLoader(
        inference_dataset,
        batch_size=config.shared.batch_size,
        shuffle=False,
        num_workers=config.shared.num_workers,
        pin_memory=True
    )
    
    # load dataframe with file names
    df_files = pd.read_csv(os.path.join(config.inference.src_inf_root, config.inference.inference_file), delimiter=';')

    # Inference loop
    all_predictions = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(inferenceDataLoader, smoothing=0.9, desc="Classifying", disable=not config.inference.verbose)):
            grid = batch['data'].to(device)
        
            pred = model(grid)
            pred = torch.nn.functional.softmax(pred, dim=1)
            pred_class = pred.argmax(dim=1)

            batch_start = batch_idx * config.shared.batch_size

            # process each sample
            for i, pred_val in enumerate(pred_class.tolist()):

                row = df_files.iloc[batch_start + i]
                # remove ".pickle" → get original pcd filename
                #original_filename = row["data"].replace(".pickle", "")
                original_filename = row["data"]
                # record prediction
                all_predictions.append([original_filename, pred_val])

    # --- Save Result CSV ---
    out_path = os.path.join(config.inference.src_inf_results, "results.csv")
    pd.DataFrame(all_predictions, columns=["file_name", "class_id"]).to_csv(out_path, sep=";", index=False)

    print("\nInference completed.")


def main(config):
    # create results directory
    version = 0
    results_dir = os.path.join(config.inference.src_inf_root, config.inference.src_inf_results)
    while os.path.exists(results_dir + f'_{version}'):
        version += 1
    results_dir = results_dir + f'_{version}'
    os.makedirs(results_dir, exist_ok=True)
    config.inference.src_inf_results = results_dir
    print(f"Results will be saved to: {results_dir}")
    
    # run inference
    start = time.time()
    inference(config)
    end_time = time.time()

    # print duration
    duration = end_time - start
    n_hours = int(duration / 3600)
    n_min = int((duration % 3600) / 60)
    n_sec = int(duration - n_hours * 3600 - n_min * 60)
    print("\n==============\n")
    print(f"TIME FOR IMFERENCE: {n_hours}:{n_min}:{n_sec}")


if __name__ == "__main__":
    parser = get_config_parser()
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    main(config)