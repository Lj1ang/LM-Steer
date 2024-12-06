# README for CS 421 Research Study. LM-Steer: Word Embeddings Are Steers for Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2305.12798-b31b1b.svg)](https://arxiv.org/abs/2305.12798)
[![ACL 2024 Outstanding Paper Award](https://img.shields.io/badge/ACL%202024-Outstanding%20Paper%20Award-ffcc00.svg)](https://acl2024.org/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-LM--Steer-ff69b4.svg)](https://huggingface.co/spaces/Glaciohound/LM-Steer)

Official Code Repository for the paper ["**LM-Steer: Word Embeddings Are Steers for Language Models**"](https://arxiv.org/abs/2305.12798) (**ACL 2024 Outstanding Paper Award**) by Chi Han, Jialiang Xu, Manling Li, Yi Fung, Chenkai Sun, Nan Jiang, Tarek Abdelzaher, Heng Ji.

[**Live Demo**](https://huggingface.co/spaces/Glaciohound/LM-Steer) | [**Paper**](https://arxiv.org/abs/2305.12798) | [**Slides**](assets/slides.pdf) | [**Poster**](assets/poster.pdf)


## Introduction


![](assets/overview_fig.jpg)

Language models (LMs) automatically learn word embeddings during pre-training on language corpora. Although word embeddings are usually interpreted as feature vectors for individual words, their roles in language model generation remain underexplored. In this work, we theoretically and empirically revisit output word embeddings and find that their linear transformations are equivalent to steering language model generation styles. We name such steers LM-Steers and find them existing in LMs of all sizes. It requires learning parameters equal to 0.2\% of the original LMs' size for steering each style.


<img src="assets/detoxification.jpg" alt="Image 1" width="45%" style="vertical-align: top;">
On tasks such as language model detoxification and sentiment control, LM-Steers can achieve comparable or superior performance compared with state-of-the-art controlled generation methods while maintaining a better balance with generation quality.

<p align="center">
  <img src="assets/dimensions.jpg" alt="Image 1" width="65%">
  <img src="assets/keywords.jpg" alt="Image 2" width="34%">
</p>

The learned LM-Steer serves as a lens in text styles: it reveals that word embeddings are interpretable when associated with language model generations, and can highlight text spans that most indicate the style differences.

<img src="assets/switch_transfer.jpg" alt="Image 1" width="65%">

A LM-Steer is transferrable between different language models by an explicit-form calculation.

<p align="center">
  <img src="assets/linear.jpg" alt="Image 1" width="45%">
  <img src="assets/compositional.jpg" alt="Image 2" width="45%">
</p>


One can also continuously steer LMs simply by scaling the LM-Steer, or compose multiple LM-Steers by adding their transformations.


## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)
  - [1. Preparing Data](#1-preparing-data)
  - [2. Training and Evaluation](#2-training-and-evaluation)
    - [2.1. LM-Steer for Detoxification](#21-lm-steer-for-detoxification)
    - [2.2. LM-Steer for Sentiment Control](#22-lm-steer-for-sentiment-control)
  - [3. Other Analytical Experiments](#3-other-analytical-experiments)
    - [3.1. LM-Steer Interpretation](#31-lm-steer-interpretation)
    - [3.2. LM-Steer Transfer](#32-lm-steer-transfer)
    - [3.3. LM-Steer Composition and Continuous Steering](#33-lm-steer-composition-and-continuous-steering)
- [Citation](#citation)


## Requirements

```
kaggle
torch
transformers
datasets
numpy
pandas
googleapiclient
```


## Usage

### 1. Preparing Data

Following the setting in [MuCoLa](https://arxiv.org/abs/2205.12558),
we download the training data from Kaggle toxic comment classification challenge.
We use prompts from MuCoLa's code repository
(placed under `data/prompts`),
which contains prompts for sentiment control and toxicity removal.

Commands for acquiring training data
(you need to setup a Kaggle account and configure the Kaggle API key):
```
# training data 
kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification
unzip jigsaw-unintended-bias-in-toxicity-classification.zip -d data/toxicity/jigsaw-unintended-bias-in-toxicity-classification
rm jigsaw-unintended-bias-in-toxicity-classification.zip

# processing
bash data/toxicity/toxicity_preprocess.sh \
    data/toxicity/jigsaw-unintended-bias-in-toxicity-classification
```



### 2. Training and Evaluation


#### 2.1. LM-Steer for Detoxification

Using GPT2-Large as the base model, we train a LM-Steer for detoxification.

```

TRIAL=detoxification-gpt2-large
mkdir -p logs/$TRIAL
PYTHONPATH=. python experiments/training/train.py \
    --dataset_name toxicity \
    --data_dir data/toxicity/jigsaw-unintended-bias-in-toxicity-classification \
    --ckpt_name logs/$TRIAL/checkpoint.pt \
    --model gpt2-large --cuda \
    --adaptor_class multiply --num_steers 2 --dummy_steer 1 --rank 1000 \
    --batch_size 32 --max_length 256 \
    --n_steps 1000 --lr 1e-2

PYTHONPATH=. python experiments/training/generate.py \
    --eval_file data/prompts/nontoxic_prompts-10k.jsonl \
    --output_file logs/$TRIAL/predictions.jsonl \
    --ckpt_name logs/$TRIAL/checkpoint.pt \
    --model gpt2-large --cuda \
    --adaptor_class multiply --num_steers 2 --rank 1000 \
    --max_length 256 --verbose --steer_values 5 1

```

The prediction file will be saved at `logs/$TRIAL/predictions.jsonl`.
We can evaluate the predictions using the following command.
To evaluate with the Perspective API from google cloud, you need to set the `export GOOGLE_API_KEY=xxxxxxx` environment variable.
Otherwise, you can remove the "toxicity" metric from the evaluation script.

```

python experiments/evaluation/evaluate.py \
    --generations_file logs/$TRIAL/predictions.jsonl \
    --metrics toxicity,ppl-big,dist-n \
    --output_file result_stats.txt
echo "Detoxification results:"
cat logs/$TRIAL/result_stats.txt

```

The evaluation script will output the evaluation results to `logs/$TRIAL/result_stats.txt`.
