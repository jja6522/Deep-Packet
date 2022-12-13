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

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize


def classification_metrics(data_path, model, n_classes, label_names=None):
    data_path = Path(data_path)
    model.eval()

    cmatrix = np.zeros((n_classes, n_classes), dtype=np.float)

    dataset_dict = datasets.load_dataset(str(data_path.absolute()))
    dataset = dataset_dict[list(dataset_dict.keys())[0]]
    try:
        num_workers = multiprocessing.cpu_count()
    except:
        num_workers = 1

    print("Test set size:", len(dataset))

    dataloader = DataLoader(dataset, batch_size=4096, num_workers=num_workers, collate_fn=dataset_collate_function)

    start = time.time()
    predicted_scores, predicted_labels, ground_truth_labels = [], [], []
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for batch_id, batch in enumerate(dataloader):
            features = batch['feature'].float().to(model.device)
            labels = batch['label'].long().to(model.device)
            outputs = model(features)

            y_hat = torch.argmax(F.log_softmax(outputs, dim=1), dim=1)

            _, predicted = torch.max(outputs, 1)
            sm = torch.nn.Softmax(dim=1)
            scores = sm(outputs)

            n_correct += (predicted == labels).sum().item()
            n_total += labels.shape[0]

            predicted_scores.append(scores.cpu().detach().numpy())
            predicted_labels.append(predicted.cpu().detach().numpy())
            ground_truth_labels.append(labels.cpu().detach().numpy())

            for i in range(len(labels)):
                cmatrix[labels[i], y_hat[i]] += 1

    print("Total test time:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start)))

    # Compute precision and recall from labels
    y_true = np.concatenate(ground_truth_labels).ravel()
    y_pred = np.concatenate(predicted_labels).ravel()

    df_metrics = pd.DataFrame(index=[*range(n_classes)],
                              columns=['label', 'precision', 'recall', 'f1_score'])

    # Average is none to get the score for each class
    df_metrics['label'] = label_names
    df_metrics['precision'] = precision_score(y_true, y_pred, average=None)
    df_metrics['recall'] = recall_score(y_true, y_pred, average=None)
    df_metrics['f1_score'] = f1_score(y_true, y_pred, average=None)

    df_metrics.loc[len(df_metrics)] = ['Wtd. Average',
                                       recall_score(y_true, y_pred, average='weighted'),
                                       precision_score(y_true, y_pred, average='weighted'),
                                       f1_score(y_true, y_pred, average='weighted')]

    # Binarize the predictions and scores
    y_bin = label_binarize(y_true , classes=[*range(n_classes)])
    y_score = np.concatenate(predicted_scores).ravel().reshape(-1, n_classes)

    # Compute average precision (AP) from prediction scores
    # AP: weighted mean of precisions achieved at each threshold
    precision_prob = dict()
    recall_prob = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision_prob[i], recall_prob[i], _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_bin[:, i], y_score[:, i])

    return cmatrix, df_metrics, recall_prob, precision_prob, average_precision

