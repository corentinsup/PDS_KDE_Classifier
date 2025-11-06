import os
import pandas as pd
import shutil

from tqdm import tqdm
from src.dataset import ModelTreesDataLoader
from torch.utils.data import DataLoader
from src.utils import *
from models.model import KDE_cls_model
from time import time
from packaging import version
from config.config import *


def inference_by_chunk(config):
    verbose = config.inference.verbose
    lst_files = os.listdir(os.path.join(config.inference.src_inf_root, config.inference.src_inf_data))

    if config.inference.chunk_size > 1 and config.inference.chunk_size < len(lst_files):
        # creates chunks of samples to infer on
        lst_chunk_of_tiles = [lst_files[x:min(y,len(lst_files))] for x, y in zip(
            range(0, len(lst_files) - config.inference.chunk_size, config.inference.chunk_size),
            range(config.inference.chunk_size, len(lst_files), config.inference.chunk_size),
            )]
        if lst_chunk_of_tiles[-1][-1] != lst_files[-1]:
            lst_chunk_of_tiles.append(lst_files[(len(lst_chunk_of_tiles)*config.inference.chunk_size)::])
        
        # creates results architecture
        if os.path.exists(os.path.join(config.inference.src_inf_root, config.inference.src_inf_results)):
            print('A "results" directory already exists.')
            answer = None
            while answer not in ['y', 'yes', 'n', 'no', '']:
                answer = input("Do you want to overwrite it (y/n)?")
                if answer.lower() in ['y', 'yes', '']:
                    shutil.rmtree(os.path.join(config.inference.src_inf_root, config.inference.src_inf_results))
                elif answer.lower() in ['n', 'no']:
                    print("Stoping the process..")
                    quit()
                else:
                    print("wrong input.")
        os.makedirs(os.path.join(config.inference.src_inf_root, config.inference.src_inf_results))

        # modify config for inference
        base_results = config.inference.src_inf_results
        base_data = config.inference.src_inf_data
        # temporarily modify config for inference
        config.inference.src_inf_data = "temp_chunk_data"
        config.inference.src_inf_results = "results_temp"

        df_results = pd.DataFrame(columns=['file_name', 'class'])
        df_failed_samples = pd.DataFrame(columns=['Index', 'data', 'label'])
        for num_chunk, chunk in tqdm(enumerate(lst_chunk_of_tiles), total=len(lst_chunk_of_tiles), desc="Infering on chunks", smoothing=0.9):
            if verbose:
                print(f"=== PROCESSING CHUNK {num_chunk + 1} / {len(lst_chunk_of_tiles)}")
                
            # create temp folder for chunks
            if os.path.exists(os.path.join(config.inference.src_inf_root, 'temp_chunk_data')):
                shutil.rmtree(os.path.join(config.inference.src_inf_root, 'temp_chunk_data'))
            os.makedirs(os.path.join(config.inference.src_inf_root, 'temp_chunk_data'))

            # copy chunk of tiles
            if verbose:
                print("Copying:")
            for _, file in tqdm(enumerate(chunk), total=len(chunk), desc="Copying", disable=not verbose):
                shutil.copyfile(
                    os.path.join(config.inference.src_inf_root, base_data, file),
                    os.path.join(config.inference.src_inf_root, config.inference.src_inf_data, file),
                )

            # call inference
            inference(config, verbose=False)
                        
            # transfert results
            for r,_,f in os.walk(os.path.join(config.inference.src_inf_root, config.inference.src_inf_results)):
                for file in f:
                    if file.endswith('.pcd'):
                        source_file_path = os.path.join(r, file)
                        rel_path = os.path.relpath(source_file_path, os.path.join(config.inference.src_inf_root, config.inference.src_inf_results))
                        target_file_path = os.path.join(config.inference.src_inf_root, base_results, rel_path)
                        os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
                        shutil.copyfile(source_file_path, target_file_path)
                    elif file.endswith('.csv'):
                        if file == 'failed_data.csv':
                            df_failed_samples = pd.concat([df_failed_samples, pd.read_csv(os.path.join(r, file), sep=';')], axis=0)
                            df_failed_samples.to_csv(os.path.join(config.inference.src_inf_root, base_results, 'failed_data.csv'), sep=';', index=False)
                        elif file == "results.csv":
                            df_results = pd.concat([df_results, pd.read_csv(os.path.join(r, file), sep=';')], axis=0)
                            df_results.to_csv(os.path.join(config.inference.src_inf_root, base_results, 'results.csv'), sep=';', index=False)
                    else:
                        print("WARNNING: Weird file: ", os.path.join(r, file))
            
            # empty temp results
            shutil.rmtree(os.path.join(config.inference.src_inf_root, config.inference.src_inf_results))
        # empty temp data
        shutil.rmtree(os.path.join(config.inference.src_inf_root, 'temp_chunk_data'))
    else:
        inference(config)


def inference(config, verbose=True):
    # create the folders for results
    if os.path.exists(os.path.join(config.inference.src_inf_root, config.inference.src_inf_results)):
        print('A "results" directory already exists.')
        answer = None
        while answer not in ['y', 'yes', 'n', 'no', '']:
            answer = input("Do you want to overwrite it (y/n)?")
            if answer.lower() in ['y', 'yes', '']:
                shutil.rmtree(os.path.join(config.inference.src_inf_root, config.inference.src_inf_results))
            elif answer.lower() in ['n', 'no']:
                print("Stoping the process..")
                quit()
            else:
                print("wrong input.")
    os.makedirs(os.path.join(config.inference.src_inf_root, config.inference.src_inf_results))

    # load the model
    if verbose:
        print("Loading model...")
    conf = {
        "num_class": config.shared.num_class,
        "grid_dim": config.shared.grid_size,
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KDE_cls_model(conf).to(device)

    if version.parse(torch.__version__) >= version.parse("2.1.0"):
        checkpoint = torch.load(config.inference.src_model, weights_only=False)
    else:
        checkpoint = torch.load(config.inference.src_model)
    # checkpoint = torch.load(SRC_MODEL, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    shape_names_path = os.path.join(config.inference.src_inf_root, 'modeltrees_shape_names.txt')

    with open(shape_names_path, 'r') as f:
        sample_labels = f.read().splitlines()

    results_root = os.path.join(config.inference.src_inf_root, config.inference.src_inf_results)

    for cls in sample_labels:
        os.makedirs(os.path.join(results_root, cls), exist_ok=True)

    # store relation between number and class label
    dict_labels = {idx: cls for idx, cls in enumerate(sample_labels)}
    
    # preprocess the samples
    if config.inference.do_preprocess:
        lst_files_to_process = [os.path.join(config.inference.src_inf_data, cls) for cls in os.listdir(os.path.join(config.inference.src_inf_root, config.inference.src_inf_data)) if cls.endswith('.pcd')]
        df_files_to_process = pd.DataFrame(lst_files_to_process, columns=['data'])
        df_files_to_process['label'] = 0
        df_files_to_process.to_csv(config.inference.src_inf_root + config.inference.inference_file, sep=';', index=False)

    # make the predictions
    if verbose:
        print("making predictions...")
    kde_transform = ToKDE(config.shared.grid_size, config.shared.kernel_size, config.shared.num_repeat_kernel)
    inferenceSet = ModelTreesDataLoader(config.inference.inference_file, config.inference.src_inf_root, split='inference', transform=None, do_update_caching=config.inference.do_preprocess, kde_transform=kde_transform, result_dir=config.inference.src_inf_results, verbose=verbose)
    if len(inferenceSet.num_fails) > 0:
        os.makedirs(os.path.join(os.path.join(config.inference.src_inf_root, config.inference.src_inf_results), 'failures/'), exist_ok=True)
        for _, file_src in inferenceSet.num_fails:
            shutil.copyfile(
                src=file_src, 
                dst=os.path.join(os.path.join(config.inference.src_inf_root, config.inference.src_inf_results), 'failures/', os.path.basename(file_src)))

    inferenceDataLoader = DataLoader(inferenceSet, batch_size=config.shared.batch_size, shuffle=False, num_workers=config.shared.num_workers, pin_memory=True)
    df_predictions = pd.DataFrame(columns=["file_name", "class"])

    for _, data in tqdm(enumerate(inferenceDataLoader, 0), total=len(inferenceDataLoader), smoothing=0.9, desc="Classifying", disable=not verbose):
        # load the samples and labels on cuda
        grid, target, filenames = data['grid'], data['label'], data['filename']
        grid, target = grid.to(device), target.to(device)

        # compute prediction
        pred = model(grid)
        pred_choice = pred.data.max(1)[1]

        # copy samples into right result folder
        for idx, pred in enumerate(pred_choice):
            fn = os.path.basename(filenames[idx].replace('.pickle', ''))
            shutil.copyfile(
                os.path.join(config.inference.src_inf_root, config.inference.src_inf_data, fn),
                os.path.join(os.path.join(config.inference.src_inf_root, config.inference.src_inf_results), dict_labels[pred.item()], fn),
                )
            df_predictions.loc[len(df_predictions)] = [os.path.join(config.inference.src_inf_data, fn), pred.item()]

    # save results in csv file
    df_predictions.to_csv(os.path.join(os.path.join(config.inference.src_inf_root, config.inference.src_inf_results), 'results.csv'), sep=';', index=False)

    # clean temp
    inferenceSet.clean_temp()


def main(config):
    # measure time
    start = time()
    # start inference
    inference_by_chunk(config)

    # print duration
    duration = time() - start
    hours = int(duration/3600)
    mins = int((duration - 3600 * hours)/60)
    secs = int((duration - 3600 * hours - 60 * mins))
    print(duration)
    print(f"Time to process inference: {hours}:{mins}:{secs}")


if __name__ == "__main__":
    
    parser = get_config_parser()
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    main(config)

    