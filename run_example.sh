#!/bin/bash

set -x
# Prepare repository and environment
git clone https://github.com/SuperGPQA/SuperGPQA.git
cd SuperGPQA
pip install -r requirements.txt


export PYTHONPATH=$(pwd)
model_name=$1
# Run inference with openai api examples
case $model_name in
    "Doubao-1.5-pro-32k-250115")
        python infer/infer.py --config config/config_default.yaml --split SuperGPQA-all --mode zero-shot --model_name $model_name --output_dir results/zero-shot --num_worker 128 --index $2 --world_size $3
        ;;
    "DeepSeek-R1")
        python infer/infer.py --config config/config_reasoning_models.yaml --split SuperGPQA-all --mode zero-shot --model_name $model_name --output_dir results/zero-shot --num_worker 128 --index $2 --world_size $3
        ;;
    "Qwen2.5-0.5B-Instruct")
        python infer/infer.py --config config/config_default.yaml --split SuperGPQA-all --mode zero-shot --model_name $model_name --output_dir results/zero-shot --batch_size 1000 --use_accel  --index $2 --world_size $3
        ;;
    "Qwen2.5-0.5B")
        python infer/infer.py --config config/config_default.yaml --split SuperGPQA-all --mode five-shot --model_name $model_name --output_dir results/five-shot --batch_size 1000 --use_accel  --index $2 --world_size $3
        ;;
esac

python eval/eval.py --evaluate_all --excel_output --output_dir results/zero-shot --save_dir results_with_status/zero-shot --json_output
python eval/eval.py --evaluate_all --excel_output --output_dir results/five-shot --save_dir results_with_status/five-shot --json_output
