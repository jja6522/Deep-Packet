import click
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from ml.metrics import confusion_matrix, get_classification_report
from ml.utils import load_cnn_model, normalise_cm
from utils import ID_TO_APP, ID_TO_TRAFFIC


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
def main(data_path, model_path, task):

    if task == 'app':

        model = load_cnn_model(model_path = model_path, gpu=True)

        # Application classification
        app_cnn_cm = confusion_matrix(
            data_path = data_path,
            model = model,
            num_class = len(ID_TO_APP)
        )

        # transform class nums to ids
        app_labels = []
        for i in sorted(list(ID_TO_APP.keys())):
            app_labels.append(ID_TO_APP[i])

        # plot results to a file
        plot_confusion_matrix(app_cnn_cm, app_labels, 'metrics/app_cnn_confusion_matrix.pdf')

        # Calculate precision, recall
        df_metrics = get_classification_report(app_cnn_cm, app_labels)

        print(df_metrics.round(2).to_markdown(index=False))

    elif task == 'traffic':

        model = load_cnn_model(model_path = model_path, gpu=True)

        # Traffic Classification
        traffic_cnn_cm = confusion_matrix(
            data_path = data_path,
            model = model,
            num_class = len(ID_TO_TRAFFIC)
        )

        traffic_labels = []
        for i in sorted(list(ID_TO_TRAFFIC.keys())):
            traffic_labels.append(ID_TO_TRAFFIC[i])

        # plot results to a file
        plot_confusion_matrix(traffic_cnn_cm, traffic_labels, 'metrics/traffic_cnn_confusion_matrix.pdf')

        # Calculate precision, recall
        df_metrics = get_classification_report(traffic_cnn_cm, traffic_labels)

        print(df_metrics.round(2).to_markdown(index=False))

    else:
        exit('Not Support')


if __name__ == '__main__':
    main()
