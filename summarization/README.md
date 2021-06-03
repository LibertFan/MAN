# Mask Attention Networks

## Dependency
Our implementation is based on the [Fairseq-v0.8.0](https://github.com/pytorch/fairseq) and [MASS](https://github.com/microsoft/MASS).

* A [PyTorch installation](http://pytorch.org/)
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* Python version 3.6

The fairseq library we use requires PyTorch version >= 1.0.0.
Please follow the instructions here: https://github.com/pytorch/pytorch#installation.


```
pip install torch==1.0.0 
pip install fairseq==0.8.0
```

## Model

## Abstractive Summarization
| Dataset | RG-1 | RG-2 | RG-L |
|:------:|:----:|:----:|:----:|
| CNN/Daily Mail | 40.98 | 18.29 | 37.88 |
| Gigaword | 38.28 | 19.46 | 35.46 |

Evaluated by [files2rouge](https://github.com/pltrdy/files2rouge). 

# Training and Inference

The scripts for training and testing Mask Attention Network is located at `scripts` folder. 
Please refer to [MASS](https://github.com/microsoft/MASS) to preprocess and get binarized data of pretraining and summarization. 

Several usage examples are shown below.
```
# Summarization in Gigaword
## To train the model
$ bash ./scripts/summarization/train/train-man-base-gigaword.sh
## To test a checkpoint
$ bash ./scripts/summarization/test/test-man-base-gigaword.sh

```

