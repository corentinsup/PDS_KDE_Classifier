import os
import csv
import pandas as pd
import time
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from src.dataset import ModelTreesDataLoader
from torch.utils.data import DataLoader
import torchvision.transforms as T
from src.utils import *
from models.model import KDE_cls_model
from src.visualization import show_log_train, show_confusion_matrix
from src.preprocess import preprocess
from config.config import *

def train_epoch(trainDataLoader, model, optimizer, criterion, device):
    loss_tot = 0
    num_samp_tot = 0
    mean_correct = []
    model.train()
    for _, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
        grid, target = data['data'], data['label']
        grid, target = grid.to(device), target.to(device)

        # training step
        optimizer.zero_grad()
        print(grid.shape)
        pred = model(grid)
        
        # loss computation
        loss = criterion(pred, target.long())
        loss_tot += loss.item()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(grid.size()[0]))

        # backpropagation
        loss.backward()
        optimizer.step()
        num_samp_tot += grid.shape[0]
    train_acc = np.mean(mean_correct)
    train_loss = loss_tot / num_samp_tot
    return train_acc, train_loss


def test_epoch(testDataLoader, model, criterion):
    loss_tot = 0
    mean_correct = []
    pred_tot = []
    target_tot = []
    class_acc = np.zeros((config.shared.num_class, 3))
    num_samp_tot = 0
    for batch_id, data in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9):
        grid, target = data['grid'], data['label']
        grid, target = grid.cuda(), target.cuda()
        pred = model(grid)
        loss = criterion(pred, target.long())
        loss_tot += loss.item()
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item()/float(grid[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(grid.size()[0]))
        num_samp_tot += grid.size()[0]
        pred_tot.append(pred_choice.tolist())
        target_tot.append(target.tolist())
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    test_acc = np.mean(mean_correct)
    test_loss = loss_tot / num_samp_tot
    pred_tot = [item for sublist in pred_tot for item in sublist]
    target_tot = [item for sublist in target_tot for item in sublist]
    return test_acc, test_loss, class_acc, pred_tot, target_tot


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
    #print('device : ' + torch.cuda.get_device_name())
    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE")
        device = torch.device('cpu')
    else:
        print("Cuda available")
        device = torch.device('cuda')

    # load data
    trainDataLoader, testDataLoader = load_data(config)
    
    # compute class weights (for unbalanced dataset)
    if config.training.use_class_weights:
        print('Calculating weights...')
        targets = pd.read_csv(config.training.ROOT_DIR + config.training.TRAIN_FILES, delimiter=';')
        targets = targets['label'].to_numpy()

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
        with torch.no_grad():
            test_acc, test_loss, class_acc, preds_test, targets_test = test_epoch(testDataLoader, model, criterion)
        line_log.append((test_acc, class_acc, test_loss))
        line_log = [el for sublists in line_log for el in sublists]     # flatten list
        print("Testing acc : ", test_acc)
        print("Testing class acc : ", class_acc)
        print("Testing loss : ", test_loss)
        if best_test_acc < test_acc:
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
                'pred': preds_test,
                'target': targets_test,
            }
            df_conf_mat_data = pd.DataFrame(conf_mat_data)
            df_conf_mat_data.to_csv(log_root + '/confmat.csv', index=False, sep=';')
            
            with open(os.path.join(config.training.ROOT_DIR, 'modeltrees_shape_names.txt'), 'r') as f:
                SAMPLE_LABELS = f.read().splitlines()
            show_confusion_matrix(log_root, preds_test, targets_test, SAMPLE_LABELS, epoch=best_epoch)

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

def load_data(config):
    '''
    Load training and testing data
    Inputs:
    - config : configuration of the training (from config/config.py)
    Outputs:
    - trainDataLoader : DataLoader for training
    - testDataLoader : DataLoader for testing
    '''

    # data transformations
    kde_transform = ToKDE(config.shared.grid_size, config.shared.kernel_size, config.shared.num_repeat_kernel)
    data_transform = T.Compose([
        RandRotate(),
        #RandScale(kernel_size),
    ])

    # preprocess the samples
    if config.training.do_preprocess:
        print("Preprocessing data...")
        preprocess(
            source_data=config.training.ROOT_DIR,
            frac_train=0.8,
            do_augment=False
        )

    # create dataloaders
    train_dataset = ModelTreesDataLoader(
        config.training.TRAIN_FILES,
        config.training.ROOT_DIR,
        split='train',
        transform=data_transform,
        do_update_caching=config.training.do_update_caching,
        kde_transform=kde_transform,
        frac=config.training.frac_training
    )
    test_dataset = ModelTreesDataLoader(
        config.training.TEST_FILES,
        config.training.ROOT_DIR,
        split='test',
        transform=None,
        do_update_caching=config.training.do_update_caching,
        kde_transform=kde_transform,
        frac=config.training.frac_testing
    )

    trainDataLoader = DataLoader(train_dataset, batch_size=config.shared.batch_size, shuffle=True, num_workers=config.shared.num_workers, pin_memory=True)
    testDataLoader = DataLoader(test_dataset, batch_size=config.shared.batch_size, shuffle=False, num_workers=config.shared.num_workers, pin_memory=True)

    return trainDataLoader, testDataLoader

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
        old_weight = checkpoint['model_state_dict']['conv1.weight']

        # duplicate the existing single channel to have wights for 2 channels
        new_weight = old_weight.repeat(1, 2, 1, 1, 1) / 2.0  # averaged copy
        checkpoint['model_state_dict']['conv1.weight'] = new_weight

        # Load model and update its state_dict
        model.load_state_dict(checkpoint['model_state_dict'])
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
