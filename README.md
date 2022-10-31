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
* Download the train and test set I created
  at [here](https://drive.google.com/file/d/1EF2MYyxMOWppCUXlte8lopkytMyiuQu_/view?usp=sharing), or download
  the [full dataset](https://www.unb.ca/cic/datasets/vpn.html) if you want to process the data from scratch.

## Data Pre-processing

```bash
python preprocessing.py -s /path/to/CompletePcap/ -t processed_data
```

## Create Train and Test

```bash
python create_train_test_set.py -s processed_data -t train_test_data
```

## Train Model

Application Classification

```bash
python train_cnn.py -d train_test_data/application_classification/train.parquet -m model/application_classification.cnn.model -t app
```

Traffic Classification

```bash
python train_cnn.py -d train_test_data/traffic_classification/train.parquet -m model/traffic_classification.cnn.model -t traffic
```

## Test Model

Application Classification

```bash
python test_cnn.py -d train_test_data/application_classification/test.parquet -m model/application_classification.cnn.model -t app
```

Traffic Classification

```bash
python test_cnn.py -d train_test_data/traffic_classification/test.parquet -m model/traffic_classification.cnn.model -t traffic
```

## Evaluation Result

### Application Classification

Results using the pre-processed dataset at [here](https://drive.google.com/file/d/1EF2MYyxMOWppCUXlte8lopkytMyiuQu_/view?usp=sharing)

| label        |   recall |   precision |   f1-score |
|:-------------|---------:|------------:|-----------:|
| AIM Chat     | 0.64411  |   0.0436111 |  0.081691  |
| Email        | 0.925285 |   0.059836  |  0.112403  |
| Facebook     | 0.841968 |   0.901054  |  0.87051   |
| FTPS         | 0.995801 |   0.999087  |  0.997441  |
| Gmail        | 0.909599 |   0.15964   |  0.271611  |
| Hangouts     | 0.881471 |   0.994193  |  0.934445  |
| ICQ          | 0.776256 |   0.0234494 |  0.0455235 |
| Netflix      | 0.988249 |   0.972343  |  0.980232  |
| SCP          | 0.916408 |   0.911979  |  0.914188  |
| SFTP         | 0.994359 |   0.977474  |  0.985844  |
| Skype        | 0.81999  |   0.97762   |  0.891894  |
| Spotify      | 0.936082 |   0.368928  |  0.529263  |
| Vimeo        | 0.981894 |   0.963267  |  0.972491  |
| Voipbuster   | 0.977807 |   0.992433  |  0.985066  |
| Youtube      | 0.976249 |   0.934379  |  0.954855  |
| Wtd. Average | 0.901613 |   0.967639  |  0.930125  |

![Application Classification](../app_cnn_confusion_matrix_alt.pdf)

Results using the original dataset at [](https://www.unb.ca/cic/datasets/vpn.html)

| label        |   recall |   precision |   f1-score |
|:-------------|---------:|------------:|-----------:|
| AIM Chat     |     0.75 |        0.04 |       0.08 |
| Email        |     0.91 |        0.09 |       0.16 |
| Facebook     |     0.86 |        0.9  |       0.88 |
| FTPS         |     1    |        1    |       1    |
| Gmail        |     0.93 |        0.12 |       0.21 |
| Hangouts     |     0.89 |        0.99 |       0.94 |
| ICQ          |     0.91 |        0.05 |       0.1  |
| Netflix      |     0.99 |        0.96 |       0.98 |
| SCP          |     0.92 |        0.87 |       0.89 |
| SFTP         |     0.99 |        0.99 |       0.99 |
| Skype        |     0.85 |        0.95 |       0.9  |
| Spotify      |     0.97 |        0.41 |       0.58 |
| Vimeo        |     0.97 |        0.96 |       0.97 |
| Voipbuster   |     0.99 |        0.98 |       0.99 |
| Youtube      |     0.98 |        0.92 |       0.95 |
| Wtd. Average |     0.92 |        0.96 |       0.93 |

![Application Classification](../app_cnn_confusion_matrix_ori.pdf)

### Traffic Classification

Results using the pre-processed dataset at [here](https://drive.google.com/file/d/1EF2MYyxMOWppCUXlte8lopkytMyiuQu_/view?usp=sharing)

| label              |   recall |   precision |   f1-score |
|:-------------------|---------:|------------:|-----------:|
| Chat               | 0.751276 |   0.261179  |   0.387607 |
| Email              | 0.809323 |   0.0725722 |   0.1332   |
| File Transfer      | 0.948507 |   0.998027  |   0.972637 |
| Streaming          | 0.984182 |   0.968708  |   0.976384 |
| Voip               | 0.916154 |   0.988083  |   0.95076  |
| VPN: Chat          | 0.991419 |   0.422251  |   0.592257 |
| VPN: File Transfer | 0.983454 |   0.938055  |   0.960218 |
| VPN: Email         | 0.995621 |   0.840729  |   0.911643 |
| VPN: Streaming     | 0.994786 |   0.997522  |   0.996152 |
| VPN: Torrent       | 0.998551 |   0.990618  |   0.994568 |
| VPN: Voip          | 0.982553 |   0.983955  |   0.983254 |
| Wtd. Average       | 0.93988  |   0.981128  |   0.95761  |

![Traffic Classification](../traffic_cnn_confusion_matrix_alt.pdf)

Results using the original dataset at [](https://www.unb.ca/cic/datasets/vpn.html)

| label              |   recall |   precision |   f1-score |
|:-------------------|---------:|------------:|-----------:|
| Chat               |     0.55 |        0.21 |       0.3  |
| Email              |     0.94 |        0.08 |       0.15 |
| File Transfer      |     0.95 |        1    |       0.97 |
| Streaming          |     0.99 |        0.96 |       0.98 |
| Voip               |     0.91 |        0.99 |       0.95 |
| VPN: Chat          |     0.99 |        0.5  |       0.66 |
| VPN: File Transfer |     0.99 |        0.91 |       0.94 |
| VPN: Email         |     0.99 |        0.92 |       0.95 |
| VPN: Streaming     |     0.99 |        1    |       1    |
| VPN: Torrent       |     1    |        0.99 |       0.99 |
| VPN: Voip          |     0.99 |        0.98 |       0.98 |
| Wtd. Average       |     0.94 |        0.98 |       0.96 |

![Traffic Classification](../traffic_cnn_confusion_matrix_ori.pdf)

## Model Files

Download the pre-trained
models [here](https://drive.google.com/file/d/1LFrx2us11cNqIDm_yWcfMES5ypvAgpmC/view?usp=sharing).

## Elapsed Time

### Preprocessing

Code ran on AWS `c5.4xlarge`

```
7:01:32 elapsed
```

### Train and Test Creation

Code ran on AWS `c5.4xlarge`

```
2:55:46 elapsed
```

### Traffic Classification Model Training

Code ran on AWS `g5.xlarge`

```
24:41 elapsed
```

### Application Classification Model Training

Code ran on AWS `g5.xlarge`

```
7:55 elapsed
```
