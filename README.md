# Deep Packet

This a fork from the implementation at https://github.com/munhouiani/Deep-Packet

## How to Use

* Clone the project
* Create environment via conda
    * For Mac
      ```bash
      conda env create -f env_mac.yaml
      ```
    * For Linux (CPU only)
      ```bash
      conda env create -f env_linux_cpu.yaml
      ```
    * For Linux (CUDA 10.x)
      ```bash
      conda env create -f env_linux_cuda10x.yaml
      ```
* Download the full dataset from [full dataset](https://www.unb.ca/cic/datasets/vpn.html) and store it into a folder called ISCXVPN2016

## Data Pre-processing

```bash
python preprocessing.py -s /path/to/ISCXVPN2016/raw_captures -t /path/to/ISCXVPN2016/processed_captures
```

## Initial class distributions for the full dataset

Application Classification
                                                       
|app_label|  count|
|:--------|------:|
|     null|4014835|
|        0|   3923|
|        1|  68594|
|        2|2697532|
|        3|4081421|
|        4|   9781|
|        5|3920675|
|        6|   3435|
|        7| 160789|
|        8| 421883|
|        9| 682965|
|       10|3809321|
|       11|  22926|
|       12|  92998|
|       13| 843408|
|       14| 140514|

Traffic Classification
                                                     
|traffic_label|  count|
|:------------|------:|
|            0| 129568|
|            1|  68594|
|            2|6602555|
|            3| 417227|
|            4|9742221|
|            5|  54928|
|            6| 252890|
|            7|  15991|
|            8|1053724|
|            9| 269115|
|           10|2368187|

## Create Train/Test split using random under-sampling

```bash
python create_train_test_set.py -s /path/to/ISCXVPN2016/processed_captures -t /path/to/ISCXVPN2016/train_test_data
```

## Create Train/Test split using SMOTE

For reading a [small sample](https://drive.google.com/file/d/1bUBt4ILBjasQfZ17PCvEMQ7O0tHi9O5J/view?usp=share_link) (~1 min): 

```bash
python SMOTE.py /path/to/ISCXVPN2016/processed_small/*.transformed_part_0000.json.gz
```

For reading the full dataset (~15 min):

```bash
python SMOTE.py /path/to/ISCXVPN2016/processed_captures/*.transformed_part_*.json.gz
```

## Train Model

Application Classification

```bash
python train_cnn.py -d /path/to/ISCXVPN2016/train_test_data/application_classification/train.parquet -m model/application_classification.cnn.model -t app
```

Traffic Classification

```bash
python train_cnn.py -d /path/to/ISCXVPN2016/train_test_data/traffic_classification/train.parquet -m model/traffic_classification.cnn.model -t traffic
```

## Test Model

Application Classification

```bash
python test_cnn.py -d /path/to/ISCXVPN2016/train_test_data/application_classification/test.parquet -m model/application_classification.cnn.model -t app
```

Traffic Classification

```bash
python test_cnn.py -d /path/to/ISCXVPN2016/train_test_data/traffic_classification/test.parquet -m model/traffic_classification.cnn.model -t traffic
```


## Results evaluation

### Application Classification Metrics

| label        |   recall |   precision |   f1-score |
|:-------------|---------:|------------:|-----------:|
| AIM Chat     |     0.87 |        0.04 |       0.08 |
| Email        |     0.94 |        0.07 |       0.13 |
| Facebook     |     0.85 |        0.91 |       0.88 |
| FTPS         |     0.99 |        1    |       1    |
| Gmail        |     0.89 |        0.15 |       0.26 |
| Hangouts     |     0.88 |        0.99 |       0.94 |
| ICQ          |     0.88 |        0.03 |       0.07 |
| Netflix      |     0.98 |        0.96 |       0.97 |
| SCP          |     0.92 |        0.93 |       0.92 |
| SFTP         |     0.99 |        0.98 |       0.99 |
| Skype        |     0.84 |        0.97 |       0.9  |
| Spotify      |     0.96 |        0.37 |       0.53 |
| Vimeo        |     0.98 |        0.95 |       0.96 |
| Voipbuster   |     0.99 |        0.97 |       0.98 |
| Youtube      |     0.98 |        0.9  |       0.94 |
| Wtd. Average |     0.91 |        0.97 |       0.93 |


![Application Classification](../../metrics/app_cnn_confusion_matrix.pdf)

### Traffic Classification metrics

| label              |   recall |   precision |   f1-score |
|:-------------------|---------:|------------:|-----------:|
| Chat               |     0.39 |        0.35 |       0.37 |
| Email              |     0.96 |        0.05 |       0.09 |
| File Transfer      |     0.94 |        1    |       0.97 |
| Streaming          |     0.96 |        0.98 |       0.97 |
| Voip               |     0.87 |        1    |       0.93 |
| VPN: Chat          |     0.97 |        0.22 |       0.36 |
| VPN: File Transfer |     0.97 |        0.83 |       0.9  |
| VPN: Email         |     0.99 |        0.52 |       0.68 |
| VPN: Streaming     |     0.99 |        1    |       1    |
| VPN: Torrent       |     0.99 |        1    |       1    |
| VPN: Voip          |     0.96 |        0.94 |       0.95 |
| Wtd. Average       |     0.91 |        0.98 |       0.94 |

![Traffic Classification](../../metrics/traffic_cnn_confusion_matrix.pdf)
