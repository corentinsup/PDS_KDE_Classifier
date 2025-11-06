import os
import time
import csv
import pandas as pd
import copy
from train import training
from src.visualization import show_log_train, show_grid_search
from config import FullConfig, SharedConfig, TrainingConfig

def main():
    # grid-search parameters
    lst_kernel_sizes = range(1, 3)
    lst_repeat_kernel = range(1, 3)

    # base args for training
    base_shared = SharedConfig(batch_size=12, num_workers=12, num_class=3, kernel_size=64, num_repeat_kernel=1, grid_size=64)
    base_training = TrainingConfig(
        num_epoch=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        do_update_caching=True,
        do_preprocess=False,
        frac_training=0.01,
        frac_testing=0.01,
        do_continue_from_existing_model=False,
        ROOT_DIR='data/modeltrees_12000/',
        TRAIN_FILES='modeltrees_train.csv',
        TEST_FILES='modeltrees_test.csv'
    )

    # create folder and file for initial logs:
    log_gridsearch_root = './log/grid_train_0'
    os.makedirs(log_gridsearch_root)

    # dataframe to save results of grid-search
    df_res = pd.DataFrame(columns=['kernel_size', 'number_repeat', 'overall_acc', 'average_acc'])
    total_time = time.time()

    # grid-search
    for kernel_size in lst_kernel_sizes:
        for repeat_kernel in lst_repeat_kernel:
            print("========================================================")
            print(f"========== TRAINING WITH {repeat_kernel} KERNELS OF SIZE {kernel_size} ==========")
            print("========================================================")
            shared = copy.deepcopy(base_shared)
            shared.kernel_size = kernel_size
            shared.num_repeat_kernel = repeat_kernel

            full_config = FullConfig(shared=shared, training=base_training, inference=None)

            # create folder for this training session
            log_file_root = os.path.join(log_gridsearch_root, f'ks={kernel_size}_rk={repeat_kernel}')
            os.makedirs(log_file_root)
            log_file = os.path.join(log_file_root, 'logs.csv')
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(['train_acc', 'train_loss', 'test_acc', 'test_class_acc', 'test_loss'])
            
            # Training
            start_time = time.time()
            best_acc, best_class_acc = training(full_config, log_file, log_file_root)
            end_time = time.time()
            
            # saving results in dataframe
            df_res.loc[len(df_res)] = [kernel_size, repeat_kernel, best_acc, best_class_acc]

            # show training logs
            show_log_train(log_file, log_file_root, do_save=True, do_show=False)

            # print time of training
            delta_time = end_time - start_time
            n_hours = int(delta_time / 3600)
            n_min = int((delta_time % 3600) / 60)
            n_sec = int(delta_time - n_hours * 3600 - n_min * 60)
            print("\n==============\n")
            print(f"TIME TO TRAIN: {n_hours}:{n_min}:{n_sec}")

    # print total time of training
    delta_time = time.time() - total_time
    n_hours = int(delta_time / 3600)
    n_min = int((delta_time % 3600) / 60)
    n_sec = int(delta_time - n_hours * 3600 - n_min * 60)
    print("\n==============\n")
    print(f"TIME TO GRID SEARCH: {n_hours}:{n_min}:{n_sec}")

    # save results of grid search
    df_res.to_csv(log_gridsearch_root + 'log_grid_search.csv', sep=';', index=False)
    show_grid_search(
        log_gridsearch_root,
        df_res[['kernel_size', 'number_repeat', 'overall_acc']],
        'Overall Accuracy',
        do_save=True,
        do_show=False,
    )
    show_grid_search(
        log_gridsearch_root,
        df_res[['kernel_size', 'number_repeat', 'average_acc']],
        'Average Accuracy',
        do_save=True,
        do_show=False,
    )


if __name__ == '__main__':
    main()
