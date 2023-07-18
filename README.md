# Contradiction-in-Peer-Review

The efficacy of the scientific publishing enterprise fundamentally rests on the strength of the peer review process. The expert reviewers' assessments significantly influence an editor's/chair's final decision. Editors/Chairs identify disagreements and varied opinions among reviewers and try to reach a consensus to make a fair and informed decision on the fate of a submission. However, with the escalating number of submissions requiring review, the editor experiences a significant workload. Here in this work, we introduce a novel task of automatically identifying contradictions between reviewers' comments on the same article. To this end, we introduce ContraSciView, a review-pair contradiction dataset based on around 8.5k papers(with around 28k review pairs containing nearly 50k review pair comments) from the open-access ICLR and NeurIPS conference reviews. We then establish a baseline model that detects contradictory comments from review pairs. Our investigations reveal both opportunities and challenges in detecting contradictions in peer reviews.

![EMNLP_contradiction (4)](https://github.com/sandeep82945/Contradiction-in-Peer-Review/assets/25824607/6464d570-d243-4567-853d-6055e42ab575)

## Repository Structure

- `data/`: This directory contains the annotated datasets used in the research. 
To download ANLI MNLI SNLI dataset open https://github.com/facebookresearch/anli and download and keep in the data folder.
- `code/`: This directory contains all the code used to analyze the data and generate the results. The code is broken down into multiple scripts according to their functionality.

## Requirements

The code for this project was written in Python 3. The required packages are listed in `requirements.txt` and can be installed with pip:

```bash
pip install -r requirements.txt

## Usage

Here's a step-by-step guide on how to use this repository:

### 1. Clone the Repository

First, clone this repository to your local machine using the following command in your terminal:
### 2. Change directory to the cloed repository
cd Contradiction-in-Peer-Review

### 2. Train the Model using the below command

python3 src/training_scratch3.py \
    --model_class_name "xlnet-large" \
    -n 1 \
    -g 1 \
    --single_gpu \
    -nr 0 \
    --max_length 280 \
    --gradient_accumulation_steps 1 \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 16 \
    --save_prediction \
    --train_data snli_train:none,mnli_train:none \
    --train_weights 1,1 \
    --eval_data snli_dev:none \
    --eval_frequency 2000 \
    --experiment_name "xlnet-large(Our)" \
    --epochs 10

Alternatively, You can run the below command for training:
sh train.sh

You can change the model to another model( Example changing the xlnet-large to roberta-large )
The trained weights will be saved in a folder named xlnet-large(Our)

We will release the pre-trained weights upon acceptance

