# Deep Packet

This a fork from the original implementation at https://github.com/munhouiani/Deep-Packet

## Below is the list of files that were modified for the experiment:

* SMOTE Implementation: [create_train_test_set.py](create_train_test_set.py)

* Train/Test data reporting: [data_reports.py](data_reports.py)

* Test and collect metrics to evaluate model performance: [test_cnn.py](test_cnn.py)

* Precision-Recall curves: [ml/metrics.py](ml/metrics.py)

## To setup the project on an ICL6 machine

1. Create an environment via conda

* For Linux (CUDA 10.6)

    ```bash
    conda env create -f env_linux_cuda116.yaml
    ```

2. Download the pre-processed dataset from [small dataset](https://drive.google.com/file/d/1bUBt4ILBjasQfZ17PCvEMQ7O0tHi9O5J/view?usp=share_link)

3. Create a directory called `processed_small` and extract the contents of the downloaded dataset

    ```bash
    mkdir processed_small
    tar -xvzf processed_small.tar.gz -C processed_small
    ```

## Create Train/Test split using Random Under-Sampling (baseline)

```python
python create_train_test_set.py --source ~/datasets/processed_small --train ~/datasets/undersampled_train_split --test ~/datasets/test_split --class_balancing under_sampling
```

## Create Train/Test split using SMOTE and Random Under-Sampling (experiment)

* Minority classes (c): 2
* Nearest Neighbors (k): 5
* Amount of SMOTE (n): 1, 2, 3, 4, 5

```python
python create_train_test_set.py --source ~/datasets/processed_small --train ~/datasets/smote_c2_n2_k5_train_split --test ~/datasets/test_split --class_balancing SMOTE+under_sampling -c 2 -n 2 -k 5 -t app --skip_test 1
```

## Train Model

Application Classification

```python
python train_cnn.py -d ~/datasets/smote_c2_n1_k5_train_split/application_classification/train.parquet -m model/application_classification.cnn.model.smote.c2n1k5 -t app
```

## Test Model

Application Classification

```python
python test_cnn.py -d ~/datasets/test_split/application_classification/test.parquet -m model/application_classification.cnn.model.smote.c2n1k5 -t app -p c2n1k5
```

## (Optional) Data reporting script to show the label distribution for any train/test split

```python
python data_reports.py -p /path/to/datasets/test_split/application_classification/test.parquet -t app -o app_test_data_dist.png
```

## (Optional) Data Pre-processing script for raw pcap files.

```python
python preprocessing.py -s /path/to/pcap_files -t /path/to/datasets/processed_new
```
