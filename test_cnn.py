import click
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ml.metrics import classification_metrics
from ml.utils import load_cnn_model, normalise_cm
from utils import ID_TO_APP, ID_TO_TRAFFIC

from sklearn.metrics import PrecisionRecallDisplay
from itertools import cycle
import matplotlib.colors as mcolors


def plot_precision_recall_curve(recall_thresholds, precision_thresholds, class_precision, class_recall, class_f1,
                                average_precision, n_classes, label_names, file_name=None):

    colors = cycle(mcolors.TABLEAU_COLORS)

    sns.set(rc={'figure.figsize': (8, 8)})
    sns.set_context('notebook', font_scale=1.15)
    fig, ax = plt.subplots()

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    for i, color in zip(range(n_classes), colors):
        display = PrecisionRecallDisplay(
            recall=recall_thresholds[i],
            precision=precision_thresholds[i]
        )

        name = '%-12s: (Pr = %-3s, Rc = %-3s, F1 = %-3s) ' % (label_names[i], round(class_precision[i], 2), round(class_recall[i], 2), round(class_f1[i], 2))
        display.plot(ax=ax, name=name, color=color)

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    wtd_avg = '%-12s: (Pr = %-3s, Rc = %-3s, F1 = %-3s) ' % ('Wtd. Average', round(class_precision[n_classes], 2), round(class_recall[n_classes], 2), round(class_f1[n_classes], 2))
    labels.extend([wtd_avg])
    labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title("Multi-Class Precision(Pr)-Recall(Rc) curves / F1-Score")

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_confusion_matrix(cm, labels, file_name=None):
    normalised_cm = normalise_cm(cm)
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(
        data=normalised_cm, cmap='YlGnBu',
        xticklabels=labels, yticklabels=labels,
        annot=True, ax=ax, fmt='.2f'
    )
    ax.set_xlabel('Predict labels')
    ax.set_ylabel('True labels')
    plt.savefig(file_name)
    plt.close()


@click.command()
@click.option('-d', '--data_path', help='testing data dir path containing parquet files', required=True)
@click.option('-m', '--model_path', help='trained model path', required=True)
@click.option('-t', '--task', help='classification task. Option: "app" or "traffic"', required=True)
@click.option('-p', '--prefix', help='prefix to append to the classifcation graphs', required=True)
def main(data_path, model_path, task, prefix):

    if task == 'app':

        model = load_cnn_model(model_path = model_path, gpu=True)
        n_classes = len(ID_TO_APP)

        # get the label names for applications
        label_names = []
        for i in sorted(list(ID_TO_APP.keys())):
            label_names.append(ID_TO_APP[i])

        # generate the classification metrics
        app_cnn_cm, class_metrics, recall_thresholds, precision_thresholds, average_precision = classification_metrics(data_path, model, n_classes, label_names)

        # print the precision, recall, f1_score per class and weighted averages
        print(class_metrics.round(2).to_markdown(index=False))

        # plot the confusion matrix in a file
        plot_confusion_matrix(app_cnn_cm, label_names, 
                              'images/'+ prefix + '_' +  task + '_confusion_matrix.png')

        # plot the precision-recall curves in a file
        plot_precision_recall_curve(recall_thresholds, precision_thresholds,
                                    class_metrics['precision'], class_metrics['recall'], class_metrics['f1_score'],
                                    average_precision, n_classes, label_names,
                                    'images/'+ prefix + '_' +  task + '_precision_recall_curve.png')

    elif task == 'traffic':

        model = load_cnn_model(model_path = model_path, gpu=True)
        n_classes = len(ID_TO_TRAFFIC)

        # get the label names for traffic categories
        label_names = []
        for i in sorted(list(ID_TO_TRAFFIC.keys())):
            label_names.append(ID_TO_TRAFFIC[i])

        # generate the classification metrics
        traffic_cnn_cm, df_metrics, recall_prob, precision_prob, average_precision = classification_metrics(data_path, model, n_classes, label_names)

        # print the precision, recall, f1_score per class and weighted averages
        print(df_metrics.round(2).to_markdown(index=False))

        # plot the confusion matrix in a file
        plot_confusion_matrix(traffic_cnn_cm, label_names,
                              'images/'+ prefix + '_' +  task + '_confusion_matrix.png')

        # plot the precision-recall curves in a file
        plot_precision_recall_curve(recall_prob, precision_prob, average_precision, n_classes, label_names,
                                    'images/'+ prefix + '_' +  task + '_precision_recall_curve.png')

    else:
        exit('Not Support')


if __name__ == '__main__':
    main()
