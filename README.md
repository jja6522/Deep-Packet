# Deep Packet

Details in blog
post: https://blog.munhou.com/2020/04/05/Pytorch-Implementation-of-Deep-Packet-A-Novel-Approach-For-Encrypted-Tra%EF%AC%83c-Classi%EF%AC%81cation-Using-Deep-Learning/

## EDIT: 2022-09-27

* Update dataset and model
* Update dependencies
* Add more data to `chat`, `file_transfer`, `voip`, `streaming` and `vpn_voip`
* Remove tor and torrent related data as they are no longer available

## EDIT: 2022-01-18

* Update dataset and model

## EDIT: 2022-01-17

* Update code and model
* Drop `petastorm`, use huggingface's `datasets` instead for data loader

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
    * For Linux (CUDA 10.2)
      ```bash
      conda env create -f env_linux_cuda102.yaml
      ```
    * For Linux (CUDA 11.3)
      ```bash
      conda env create -f env_linux_cuda113.yaml
      ```
    * For Linux (CUDA 11.6)
      ```bash
      conda env create -f env_linux_cuda116.yaml
      ```
* Download the full dataset from [full dataset](https://www.unb.ca/cic/datasets/vpn.html) and store it into a folder called ISCXVPN2016

## Data Pre-processing

```bash
python preprocessing.py -s /path/to/ISCXVPN2016/raw_captures -t /path/to/ISCXVPN2016/processed_captures
```

## Create Train and Test

```bash
python create_train_test_set.py -s /path/to/ISCXVPN2016/processed_captures -t /path/to/ISCXVPN2016/train_test_data
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

## Evaluation Result

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
