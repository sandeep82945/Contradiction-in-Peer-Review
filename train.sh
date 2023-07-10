#!/bin/bash
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
