<h2 align="center" style="font-size: 2.5em; font-weight: bold; color: #2c3e50;">
  <i>SuperGPQA</i>: Scaling LLM Evaluation across<br>
  285 Graduate Disciplines
</h2>

<p align="center">
  <a href="https://supergpqa.github.io/" style="margin: 0 10px;">🌐 Homepage</a> |
  <a href="https://huggingface.co/datasets/m-a-p/SuperGPQA" style="margin: 0 10px;">🤗 Paper</a> |
  <a href="TBD" style="margin: 0 10px;">📖 ArXiv</a> |
  <a href="https://huggingface.co/spaces/m-a-p/SuperGPQA" style="margin: 0 10px;">🏆 Leaderboard</a> |
  <a href="https://github.com/SuperGPQA/SuperGPQA" style="margin: 0 10px;">🐙 GitHub</a>
</p>


This repository contains the evaluation code for the paper "[SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines]()".

---

## 🔔 Introduction


<p align="center">
  <img src="images/main_final.png" alt="SuperGPQA Overview" style="width: 800px;"> 
</p>


**SuperGPQA** is a comprehensive benchmark that evaluates graduate-level knowledge and reasoning capabilities across 285 disciplines. Our benchmark employs a novel Human-LLM collaborative filtering mechanism to eliminate trivial or ambiguous questions through iterative refinement based on both LLM responses and expert feedback. Our experimental results reveal significant room for improvement in the performance of current state-of-the-art LLMs across diverse knowledge domains (e.g., the reasoning-focused model DeepSeek-R1 achieved the highest accuracy of 61.82% on **SuperGPQA**), highlighting the considerable gap between current model capabilities and artificial general intelligence. Additionally, we present comprehensive insights from our management of a large-scale annotation process, involving over 80 expert annotators and an interactive Human-LLM collaborative system, offering valuable methodological guidance for future research initiatives of comparable scope.

---

## ⚙️ Installation 

To install the required packages, run:

```bash
# Prepare repository and environment
git clone git@github.com:SuperGPQA/SuperGPQA.git
cd ./SuperGPQA
pip install -r requirements.txt
```

---

## Inference

You can directly perform inference on the selected models using the following command:
```bash
export PYTHONPATH=$(pwd)

# Local model inference
python infer/infer.py --config <CONFIG_PATH> --split <TASKS> --mode <MODE> --model_name <MODEL_NAME> --output_dir <OUTPUT_DIR> --batch_size <BATCH_SIZE> --use_accel --index <INDEX> --world_size <WORLD_SIZE>

# API model inference
python infer/infer.py --config <CONFIG_PATH> --split <TASKS> --mode <MODE> --model_name <MODEL_NAME> --output_dir <OUTPUT_DIR> --num_worker <NUM_WORKERS> --index <INDEX> --world_size <WORLD_SIZE>
```

Example:
```bash
export PYTHONPATH=$(pwd)

# Local model inference with zero-shot mode
python infer/infer.py --config config/config_default.yaml --split SuperGPQA-all --mode zero-shot --model_name Doubao-1.5-pro-32k-250115 --output_dir results/zero-shot --num_worker 128 --index 0 --world_size 1

# Local reasoning model inference with zero-shot mode
python infer/infer.py --config config/config_reasoning_models.yaml --split SuperGPQA-all --mode zero-shot --model_name DeepSeek-R1 --output_dir results/zero-shot --num_worker 128 --index 0 --world_size 1

# Local chat model inference with accelerated setup and zero-shot mode
python infer/infer.py --config config/config_default.yaml --split SuperGPQA-all --mode zero-shot --model_name Qwen2.5-0.5B-Instruct --output_dir results/zero-shot --batch_size 1000 --use_accel --index 0 --world_size 1

# Local chat model inference with five-shot mode
python infer/infer.py --config config/config_default.yaml --split SuperGPQA-all --mode five-shot --model_name Qwen2.5-0.5B --output_dir results/five-shot --batch_size 1000 --use_accel --index 0 --world_size 1
```

More examples can be found in the shell scripts of this repository. 🔗

**Parameter Explanations for Inference Script**
- config: Path to the configuration file.
- split: Specify the task categories for evaluation.
Available options include:
        - SuperGPQA-all (or any other task list depending on the model being used).
-  mode: Choose from different evaluation modes (zero-shot, five-shot). Default is to evaluate all modes.
- use_accel: Enables acceleration options for faster inference. All local model experimental results in this repository use vllm for acceleration.
- num_worker: Set the number of concurrent workers (for API calls). For local models, this should typically be set to 1.
- batch_size: Set the batch size for local model inference. This is especially useful for models that can process multiple examples at once (e.g., local chat models).
- index: The index for the data split. It determines which part of the dataset is processed.
- world_size: Defines the size of the distributed setup (useful for multi-node inference).

📝 Notes
- If the inference is unexpectedly interrupted, a temporary file .jsonl.tmp will be saved. You can directly rerun the command to resume from the last checkpoint.
- After inference is complete, check the response field in the saved JSONL file in the output_dir. This field should typically be of string type; if it is of dict type, the error field will contain error information. You can rerun the command to re-infer any issues that caused errors.

🛠️ Run Custom Model
- --model_name: This parameter must align with the filenames in the infer/models directory. You can choose from available built-in models.
- Adding a Custom Model:
        1. Create a new .py file for your model in the infer/models directory.
        2. Update the configuration in __init__.py to include the new model.
- For more details, please refer to the documentation for the specific model you want to add.

---

## ⭐ Evaluation
After completing the inference and ensuring no error messages are displayed, proceed with the answer parsing and evaluation pipeline as follows:

```bash
export PYTHONPATH=$(pwd)

# Evaluate results from zero-shot inference
python eval/eval.py --evaluate_all --excel_output --output_dir results/zero-shot --save_dir results_with_status/zero-shot --json_output

# Evaluate results from five-shot inference
python eval/eval.py --evaluate_all --excel_output --output_dir results/five-shot --save_dir results_with_status/five-shot --json_output
```


## 📫 Contact
Xinrun Du: duxinrun2000@gmail.com

Ge Zhang: zhangge.eli@bytedance.com

## 📚 Citation

**BibTeX:**
```bibtex
@misc{ma2024korbenchbenchmarkinglanguagemodels,
    title={SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines}, 
    author={SuperGPQA Team},
    year={2025},
    eprint={TBD},
    archivePrefix={arXiv},
    primaryClass={cs.DB},
    url={TBD}, 
}
```

