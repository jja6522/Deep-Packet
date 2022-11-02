import multiprocessing
from pathlib import Path

import datasets
import numpy as np
import torch
import pandas as pd
import time
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ml.dataset import dataset_collate_function


def confusion_matrix(data_path, model, num_class):
    data_path = Path(data_path)
    model.eval()

    cm = np.zeros((num_class, num_class), dtype=np.float)

    dataset_dict = datasets.load_dataset(str(data_path.absolute()))
    dataset = dataset_dict[list(dataset_dict.keys())[0]]
    try:
        num_workers = multiprocessing.cpu_count()
    except:
        num_workers = 1

    print("Test set size:", len(dataset))

    dataloader = DataLoader(dataset, batch_size=4096, num_workers=num_workers, collate_fn=dataset_collate_function)

    start = time.time()
    for batch_id, batch in enumerate(dataloader):
        x = batch['feature'].float().to(model.device)
        y = batch['label'].long()
        y_hat = torch.argmax(F.log_softmax(model(x), dim=1), dim=1)

        for i in range(len(y)):
            cm[y[i], y_hat[i]] += 1

    print("Total test time:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start)))

    return cm


def get_precision(cm, i):
    tp = cm[i, i]
    tp_fp = cm[:, i].sum()

    return tp / tp_fp


def get_recall(cm, i):
    tp = cm[i, i]
    p = cm[i, :].sum()

    return tp / p


def get_classification_report(cm, labels=None):
    rows = []
    for i in range(cm.shape[0]):
        precision = get_precision(cm, i)
        recall = get_recall(cm, i)
        f1_score = 2 * ((precision * recall) / (precision + recall))

        if labels:
            label = labels[i]
        else:
            label = i

        row = {
            'label': label,
            'recall': recall,
            'precision': precision,
            'f1-score': f1_score
        }
        rows.append(row)

    df_metrics = pd.DataFrame(rows)

    # Compute the weighted rc, pr, f1
    support_prop = cm.sum(axis=1, keepdims=True) / cm.sum()
    weighted_rc = df_metrics['recall'] @ support_prop
    weighted_pr = df_metrics['precision'] @ support_prop
    weighted_f1 = df_metrics['f1-score'] @ support_prop

    df_metrics.loc[len(df_metrics)] = ['Wtd. Average', weighted_rc.item(), weighted_pr.item(), weighted_f1.item()]

    return df_metrics

