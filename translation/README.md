# Mask Attention Networks - Machine Translation

## Requirements and Installation
Our implementation follows [Fairseq-v0.8.0](https://github.com/pytorch/fairseq) and [Macaron-net](https://github.com/zhuohan123/macaron-net).

* A [PyTorch installation](http://pytorch.org/)
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* Python version 3.6

The fairseq library we use requires PyTorch version >= 0.4.0.
Please follow the instructions here: https://github.com/pytorch/pytorch#installation.

After PyTorch is installed, you can install fairseq with:
```
pip install -r requirements.txt
python setup.py build develop
```

## Model
| Dataset | BLEU |
|:------:|:----:|
| IWSLT14 De-En | 36.3 |
| WMT14 En-De (Small)| 29.1|
| WMT14 En-De (Big)| 30.4 |

## Training and Inference

The scripts for training and testing Macaron Net is located at `macaron-scripts` folder. Please refer to [this page](translation-scripts/data-preprocessing/README.md) to preprocess and get binarized data or use the data we provided in the next section. To reproduce the results by yourself:

```
# IWSLT14 De-En
## To train the model
$ bash ./translation-scripts/train/train-iwslt.sh
## To test a checkpoint
$ bash ./translation-scripts/test/test-iwslt.sh

# WMT14 En-De base
## To train the model
$ bash ./macaron-scripts/train/train-wmt-base.sh
## To test a checkpoint
$ bash ./macaron-scripts/test/test-wmt-base.sh

# WMT14 En-De big
## To train the model
$ bash ./macaron-scripts/train/train-wmt-big.sh
## To test a checkpoint
$ bash ./macaron-scripts/test/test-wmt-big.sh
```
