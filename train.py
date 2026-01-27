import os
import csv
import pandas as pd
import time
import torchvision.transforms as T

from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from src.utils import *
from models.model import KDE_cls_model
from src.visualization import show_log_train, show_confusion_matrix
from config.config import *
from src.csv_creation import preprocess_dataset
from src.dataset import TrainPCDDataset
from src.transforms import *

def train_epoch(trainDataLoader, model, optimizer, criterion, device):
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    model.train()
    
    for _, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
        grid, target = data['data'], data['label']
        grid, target = grid.to(device), target.to(device).long()

        # training step
        optimizer.zero_grad()
        pred = model(grid)
        
        # loss computation
        loss = criterion(pred, target.long())
        
        # backpropagation
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * grid.size(0) # sum loss over batch
        pred_choice = pred.argmax(dim=1)           # get predicted class
        running_corrects += torch.sum(pred_choice == target).item()       
        total_samples += grid.shape[0]
    
    train_acc = running_corrects / total_samples
    train_loss = running_loss / total_samples
    return train_acc, train_loss


def test_epoch(testDataLoader, model, criterion, device, config):
    # set model to evaluation mode
    model.eval()

    # metrics initialization
    running_loss = 0.0
    total_samples = 0
    correct_samples = 0
    class_correct = np.zeros(config.shared.num_class)
    class_total = np.zeros(config.shared.num_class)
    
    pred_all = []
    target_all = []

    with torch.no_grad():
        for batch in tqdm(testDataLoader, total=len(testDataLoader), smoothing=0.9):
            # move data to device
            grid = batch['data'].to(device)
            target = batch['label'].to(device).long()

            # forward pass
            pred = model(grid)
            loss = criterion(pred, target)
            running_loss += loss.item() * grid.size(0)
            total_samples += grid.shape[0]

            # predictions
            pred_choice = pred.argmax(dim=1)
            correct_samples += torch.sum(pred_choice == target).item()

            # per class accuracy
            for cls in range(config.shared.num_class):
                cls_mask = (target == cls)
                class_total[cls] += cls_mask.sum().item()
                class_correct[cls] += (pred_choice[cls_mask] == cls).sum().item()

            # store predictions
            pred_all.extend(pred_choice.cpu().tolist())
            target_all.extend(target.cpu().tolist())
        
    # compute final accuracies and losses
    test_loss = running_loss / total_samples
    test_acc = correct_samples / total_samples
    class_acc = np.mean(class_correct/np.maximum(class_total, 1))  # avoid division by zero

    return test_acc, test_loss, class_acc, pred_all, target_all

def training(config, log_file, log_root):
    '''
    Main training function
    Inputs:
    - config : configuration of the training (from config/config.py)
    - log_file : csv file where to save logs
    - log_root : root folder where to save logs
    '''
    # check torch and if cuda is available
    print("torch version : " + torch.__version__)
    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE")
        device = torch.device('cpu')
    else:
        print("Cuda available")
        device = torch.device('cuda')

    # creation of the csv file with corresponding files to train/test on
    if config.training.do_preprocess:
        print("Preprocessing dataset...")
        preprocess_dataset(
            src_root = config.training.ROOT_DIR,
            mode = "train",
            frac_train = config.training.frac_training,
            output_file = None
        )

    # data transformations
    point_cloud_transforms = PointCloudTransforms(config)
    voxel_transforms = VoxelTransforms(config)

    '''kde_transform = ToKDE(config.shared.grid_size, config.shared.kernel_size, config.shared.num_repeat_kernel)
    data_transform = T.Compose([
        RandRotate(),
        #RandScale(kernel_size),
        # Adding tree crop transform for data augmentation
        TreeCropTransform(crop_ratio=config.training.crop_ratio, probability=config.training.probability),
    ])
    # Optional: creation of pickles from pcd files
    if config.training.do_update_caching:
        print("Converting PCD to Pickle for training set...")
        csv_to_pickles(
            csvfile=os.path.join(config.training.ROOT_DIR, config.training.TRAIN_FILES),
            root_dir=config.training.ROOT_DIR,
            split='train',
            kde_transform=kde_transform,
            pickle_subdir='pickles_train'
        )
        print("Converting PCD to Pickle for testing set...")
        csv_to_pickles(
            csvfile=os.path.join(config.training.ROOT_DIR, config.training.TEST_FILES),
            root_dir=config.training.ROOT_DIR,
            split='test',
            kde_transform=kde_transform,
            pickle_subdir='pickles_test'
        )
        
    # load datasets 
    trainDataLoader = makeDataloader(
                            csvfile = os.path.join(config.training.ROOT_DIR, 'modeltrees_train_pickles.csv'),
                            pickle_dir = os.path.join(config.training.ROOT_DIR, 'pickles_train'),
                            transform = data_transform,
                            batch_size = config.shared.batch_size,
                            shuffle = True,
                            num_workers = config.shared.num_workers
    )
    testDataLoader = makeDataloader(
                            csvfile = os.path.join(config.training.ROOT_DIR, 'modeltrees_test_pickles.csv'),
                            pickle_dir = os.path.join(config.training.ROOT_DIR, 'pickles_test'),
                            transform = data_transform,
                            batch_size = config.shared.batch_size,
                            shuffle = True,
                            num_workers = config.shared.num_workers
    )'''
    # loading datasets and creating pickle files
    print("Loading datasets...")
    train_dataset = TrainPCDDataset(
                        csvfile = os.path.join(config.training.ROOT_DIR, config.training.TRAIN_FILES),
                        root_dir = config.training.ROOT_DIR,
                        n_augmentations = config.training.n_augmentations,
                        points_transforms = point_cloud_transforms,
                        voxel_transforms = voxel_transforms
    )
    test_dataset = TrainPCDDataset(
                        csvfile = os.path.join(config.training.ROOT_DIR, config.training.TEST_FILES),
                        root_dir = config.training.ROOT_DIR,
                        n_augmentations = config.training.n_augmentations,
                        points_transforms = point_cloud_transforms,
                        voxel_transforms = None
    )

    trainDataLoader = DataLoader(
                            train_dataset,
                            batch_size = config.shared.batch_size,
                            shuffle = True,
                            num_workers = config.shared.num_workers,
                            pin_memory = True,
    )
    testDataLoader = DataLoader(
                            test_dataset,
                            batch_size = config.shared.batch_size,
                            shuffle = True,
                            num_workers = config.shared.num_workers,
                            pin_memory = True,
    )

    # compute class weights (for unbalanced dataset)
    if config.training.use_class_weights:
        print('Calculating weights from actual dataset distribution...')
        
        # Get original class distribution from CSV
        original_labels = train_dataset.labels
        original_counts = {}
        for label in original_labels:
            original_counts[label] = original_counts.get(label, 0) + 1
        
        print(f'Original samples per class: {original_counts}')
        print(f'With n_augmentations={config.training.n_augmentations} and crop_probability={config.training.probability}')
        
        # Calculate expected class distribution after augmentation
        # Class 0 (Garbage): always stays class 0
        # Class 1 & 2 (Multi/Single): become class 3 with probability p during crop augmentation
        p = config.training.probability
        n_aug = config.training.n_augmentations
        
        class_counts = {
            0: original_counts.get(0, 0) * n_aug,  # Garbage stays unchanged
            1: original_counts.get(1, 0) * n_aug * (1 - p),  # Multi that don't get cropped
            2: original_counts.get(2, 0) * n_aug * (1 - p),  # Single that don't get cropped
            3: (original_counts.get(1, 0) + original_counts.get(2, 0)) * n_aug * p  # Cropped trees
        }
        
        print(f'Expected class distribution with augmentation: {class_counts}')
        
        # Create target array based on expected counts
        targets = []
        for label, count in class_counts.items():
            targets.extend([label] * int(count))
        targets = np.array(targets)

        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(targets),
            y=targets,
        )
        class_weights = torch.tensor(weights, dtype=torch.float, device=device)
        print('Class weights:', class_weights)
    else:
        print('Skipping class weights (dataset assumed balanced)')
        class_weights = None

    # initialize model
    model, optimizer, scheduler, criterion = initialize_model(config, device, class_weights)

    # loop on epochs
    best_test_acc = 0
    best_test_class_acc = 0
    best_test_loss = 0
    best_epoch = 0
    for epoch in range(config.training.num_epoch):
        line_log = []

        # training
        print(f"Training on epoch {str(epoch+1)}/{str(config.training.num_epoch)}:")
        train_acc, train_loss = train_epoch(trainDataLoader, model, optimizer, criterion, device)
        scheduler.step()
        line_log.append((train_acc, train_loss))
        print("Training acc : ", train_acc)
        print("Training loss : ", train_loss)
        print("Testing...")

        # testing
        test_acc, test_loss, class_acc, preds, targets = test_epoch(testDataLoader, model, criterion, device, config)
        line_log.append((test_acc, class_acc, test_loss))
        line_log = [el for sublists in line_log for el in sublists]     # flatten list
        print("Testing acc : ", test_acc)
        print("Testing class acc : ", class_acc)
        print("Testing loss : ", test_loss)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            best_test_class_acc = class_acc
            best_test_loss = test_loss

            # save model
            print("Best results : saving model...")
            torch.save({
                'epoch': epoch,
                'batch_size': config.shared.batch_size,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': test_acc,
                'test_class_acc': class_acc,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
            }, log_root + "/model_KDE.tar")

            # save preds and create confusion matrix
            conf_mat_data = {
                'pred': preds,
                'target': targets,
            }
            df_conf_mat_data = pd.DataFrame(conf_mat_data)
            df_conf_mat_data.to_csv(log_root + '/confmat.csv', index=False, sep=';')
            
            with open(os.path.join(config.training.ROOT_DIR, 'modeltrees_shape_names.txt'), 'r') as f:
                SAMPLE_LABELS = f.read().splitlines()
            show_confusion_matrix(log_root, preds, targets, SAMPLE_LABELS, epoch=best_epoch)

        # update logs
        with open(log_file, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow([str(x) for x in line_log])

    # best results
    print("\n==============\n")
    print("BEST RESULTS ON EPOCH ", best_epoch+1)
    print("BEST TEST ACC: ", best_test_acc)
    print("BEST TEST CLASS ACC: ", best_test_class_acc)
    print("BEST TEST LOSS: ", best_test_loss)

def initialize_model(config, device, class_weights):
    '''
    Initialize the model, optimizer and loss function
    Inputs:
    - config : configuration of the training (from config/config.py)
    - class_weights : weights for each class (for unbalanced dataset)
    Outputs:
    - model : the model to train
    - optimizer : the optimizer
    - criterion : the loss function
    '''
    conf = {
        "num_class": config.shared.num_class,
        "grid_dim": config.shared.grid_size
    }
    model = KDE_cls_model(conf).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=config.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)

    # load model if needed
    if config.training.load_model:
        checkpoint = torch.load(config.training.model_path, map_location=device)
        
        pretrained = checkpoint["model_state_dict"]
        model_dict = model.state_dict()

        # adapt conv1 weights from 1 to 2 input channels
        old_w = pretrained['conv1.weight']                      # (32,1,3,3,3)
        new_w = torch.cat([old_w, old_w], dim=1)                # (32,2,3,3,3)
        pretrained['conv1.weight'] = new_w                      # update in-place
        
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
        print("Model loaded from ", config.training.model_path)

        if config.training.resume_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer resumed from ", config.training.model_path)
    else :
        print("Training model from scratch")

    return model, optimizer, scheduler, criterion

def main(config):
    # create folder for this training session
    version = 0
    while os.path.exists(f'./log/train_{version}'):
        version += 1
    log_root = f'./log/train_{version}'
    os.makedirs(log_root)

    # create CSV log file
    log_file = os.path.join(log_root, 'logs.csv')
    with open(log_file, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['train_acc', 'train_loss', 'test_acc', 'test_class_acc', 'test_loss'])

    # Training
    start_time = time.time()
    training(config, log_file, log_root)
    end_time = time.time()

    # Plots of results
    show_log_train(log_file, log_root)

    # print time of training
    delta_time = end_time - start_time
    n_hours = int(delta_time / 3600)
    n_min = int((delta_time % 3600) / 60)
    n_sec = int(delta_time - n_hours * 3600 - n_min * 60)
    print("\n==============\n")
    print(f"TIME TO TRAIN: {n_hours}:{n_min}:{n_sec}")

if __name__ == "__main__":
    parser = get_config_parser()
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    main(config)