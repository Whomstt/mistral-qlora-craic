# mistral-qlora-craic
The Mistral-QLoRA-Craic project involves fine-tuning the Ministral-3-3B-Base-2512 model using QLoRA for Irish-English dialect adaptation. A 4-bit quantization (NF4) was first applied to the base model using the HF BitsAndBytes library to provide similar model performance with a huge reduction in memory usage and latency. LoRA config parameters were tuned to ensure that sufficient model parameters were trained, while considering the GPU budget available. A suitable dataset was prepared before commencing training consisting of many Irish-English dialect samples. Model performance was evaluated by comparing the fine-tuned model against the base model on several prompts.

## Requirements
- Nvidia CUDA GPU
- Python 3.13

## Setup Guide
Create a compatible virtual environment:
- python -m venv .venv
- .venv/Scripts/activate
- pip install -r requirements.txt

## References
Core Workflow: https://pub.towardsai.net/from-quantization-to-inference-beginners-guide-for-practical-fine-tuning-52c7c3512ef6
<br>LoRA Hyperparameter Tuning: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
<br>MLflow Experiment Tracking: https://mlflow.org/docs/latest/ml/deep-learning/transformers/tutorials/fine-tuning/transformers-peft/