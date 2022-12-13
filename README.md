# Deep Packet

This a fork from the original implementation at https://github.com/munhouiani/Deep-Packet

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

## Create Train/Test split using random under-sampling (baseline)

```python
python create_train_test_set.py --source /path/to/datasets/processed_small --train /path/to/datasets/undersampled_train_split --test /path/to/datasets/test_split --class_balancing under_sampling
```

## Train Model

Application Classification

```python
python train_cnn.py -d ~/datasets/undersampled_train_split/application_classification/train.parquet -m model/application_classification.cnn.model.base -t app
```

Traffic Classification

```python
python train_cnn.py -d /path/to/datasets/undersampled_train_split/traffic_classification/train.parquet -m model/traffic_classification.cnn.model -t traffic
```

## Test Model

Application Classification

```python
python test_cnn.py -d ~/datasets/test_split/application_classification/test.parquet -m model/application_classification.cnn.model.base -t app
```

Traffic Classification

```python
python test_cnn.py -d /path/to/datasets/test_split/traffic_classification/test.parquet -m model/traffic_classification.cnn.model -t traffic
```

## (Optional) Data reporting script to show the label distribution for any train/test split

```python
python data_reports.py -p /path/to/datasets/test_split/application_classification/test.parquet -t app -o app_test_data_dist.png
```

## (Optional) Data Pre-processing script for raw pcap files.

```python
python preprocessing.py -s /path/to/pcap_files -t /path/to/datasets/processed_new
```
