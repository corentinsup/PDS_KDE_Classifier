import numpy as np
import os
import pandas as pd
from sklearn.metrics import f1_score 

def multi_class_f1_score(csv_path):
    ''' Compute the F1 score.
    Args:
        y_true: Ground truth labels, shape (N,).
        y_pred: Predicted labels, shape (N,).
    Returns:
        f1: F1 score.
    '''
    # import csv file with targets and predictions
    results_df = pd.read_csv(csv_path, sep=';')
    target_all = results_df['target'].to_numpy()
    pred_all = results_df['pred'].to_numpy()

    # compute F1 score
    f1_per_class = f1_score(target_all, pred_all, average=None)
    f1_micro = f1_score(target_all, pred_all, average='micro')
    f1_macro = f1_score(target_all, pred_all, average='macro')
    f1_weighted = f1_score(target_all, pred_all, average='weighted')

    print('classes', np.unique(target_all))
    print('F1 score per class:', f1_per_class)
    print('Micro F1 score:', f1_micro)
    print('Macro F1 score:', f1_macro)
    print('Weighted F1 score:', f1_weighted)

if __name__ == '__main__':
    multi_class_f1_score("./log/train_6/confmat.csv")
