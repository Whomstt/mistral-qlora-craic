# mistral-qlora-craic
The Mistral-QLoRA-Craic project involved fine-tuning the Ministral-3-3B-Base-2512 model using QLoRA for Irish-English dialect adaptation. A 4-bit quantization (NF4) was applied to the base model using the HF BitsAndBytes library to provide similar model performance with a huge reduction in memory usage and latency. LoRA config parameters were tuned to ensure that sufficient model parameters were trained, while considering the GPU budget available. A suitable dataset was created with many Irish-English dialect samples by combining inspirations from various sources. Evaluation was carried out by using MLflow to benchmark fine-tuned vs. base model performance to ensure reliability. Hugging Face and vLLM support was implemented to support efficient GPU inference and compatibility across devices.
## Downloads
- Base Model (HF): https://huggingface.co/mistralai/Ministral-3-3B-Base-2512
- Quantised Model (HF): https://huggingface.co/Whomstt/Ministral-3-3B-Base-2512-bnb-nf4
- LoRA Adapter (HF): https://huggingface.co/Whomstt/mistral-qlora-craic
- Dataset (HF): https://huggingface.co/datasets/Whomstt/irish-english-dialect
## Requirements
**Note vLLM hosting for GPU inference requires Linux/WSL**
- Nvidia CUDA GPU with latest drivers installed
- Python 3.13
## Setup Guide
### Windows
- python3.13 -m venv .venv
- .venv/Scripts/activate
- pip install -r requirements.txt
### Linux/WSL
- python3.13 -m venv .venv
- source .venv/bin/activate
- pip install -r requirements.txt
### CUDA
**Note system-wide install may be required**
- CUDA Toolkit 13.0 https://developer.nvidia.com/cuda-13-0-0-download-archive
- cuDNN 9.17.1 https://developer.nvidia.com/cudnn-9-17-1-download-archive
## Usage Guide
### Hugging Face
- hf_inference.py
### vLLM
- vllm_inference.py
## References
Core Workflow: https://pub.towardsai.net/from-quantization-to-inference-beginners-guide-for-practical-fine-tuning-52c7c3512ef6
<br>LoRA Hyperparameter Tuning: https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
<br>MLflow Experiment Tracking: https://mlflow.org/docs/latest/ml/deep-learning/transformers/tutorials/fine-tuning/transformers-peft/

## Dataset Inspirations:
https://tandem.net/blog/irish-slang-words-phrases
<br>https://www.reddit.com/r/ireland/comments/yraz43/irishenglish_is_the_best_english_dialect_by_a_mile/
<br>https://blog.duolingo.com/irish-slang/
<br>https://www.superprof.ie/blog/irish-english-dialect/
<br>https://stancarey.wordpress.com/2014/05/26/10-words-only-used-in-irish-english/