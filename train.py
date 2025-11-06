import os
import csv
import pandas as pd
import time
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from dataset import ModelTreesDataLoader
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import *
from models.model import KDE_cls_model
from visualization import show_log_train, show_confusion_matrix
from config.config import *


def train_epoch(trainDataLoader, model, optimizer, criterion):
    loss_tot = 0
    num_samp_tot = 0
    mean_correct = []
    model.train()
    for _, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
        grid, target = data['grid'], data['label']
        grid, target = grid.to('cuda:0'), target.to('cuda:0')
        optimizer.zero_grad()
        pred = model(grid)
        loss = criterion(pred, target.long())
        loss_tot += loss.item()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(grid.size()[0]))
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
    print('device : ' + torch.cuda.get_device_name())
    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE")
    else:
        print("Cuda available")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # transformation
    kde_transform = ToKDE(config.shared.grid_size, config.shared.kernel_size)
    data_transform = transforms.Compose([
        RandRotate(),
        #RandScale(kernel_size),
    ])

    # load datasets
    trainingSet = ModelTreesDataLoader(config.training.TRAIN_FILES, config.training.ROOT_DIR, split='train', transform=data_transform, do_update_caching=config.training.do_update_caching, kde_transform=kde_transform, frac=config.training.frac_training)
    testingSet = ModelTreesDataLoader(config.training.TEST_FILES, config.training.ROOT_DIR, split='test', transform=None, do_update_caching=config.training.do_update_caching, kde_transform=kde_transform, frac=config.training.frac_testing)

    torch.manual_seed(42)
    trainDataLoader = DataLoader(trainingSet, batch_size=config.shared.batch_size, shuffle=True, num_workers=config.shared.num_workers, pin_memory=True)
    testDataLoader = DataLoader(testingSet, batch_size=config.shared.batch_size, shuffle=True, num_workers=config.shared.num_workers, pin_memory=True)

    # get class weights:
    print('Calculating weights...')
    targets = pd.read_csv(config.training.ROOT_DIR + config.training.TRAIN_FILES, delimiter=';')
    targets = targets['label'].to_numpy()
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(targets),
        y=targets,
    )

    print('Weights : ', weights)
    class_weights = torch.tensor(weights, dtype=torch.float, device=device)

    # create model
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

    # loop on epochs
    best_test_acc = 0
    best_test_class_acc = 0
    best_test_loss = 0
    best_epoch = 0
    for epoch in range(config.training.num_epoch):
        line_log = []

        # training
        print(f"Training on epoch {str(epoch+1)}/{str(config.training.num_epoch)}:")
        train_acc, train_loss = train_epoch(trainDataLoader, model, optimizer, criterion)
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

    return best_test_acc, best_test_class_acc


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
    best_acc, best_class_acc = training(config, log_file, log_root)
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
