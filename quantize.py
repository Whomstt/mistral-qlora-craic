# Import the necessary libraries
import torch
from transformers import (
    BitsAndBytesConfig,
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
)
from datasets import load_dataset
from peft import (
    prepare_model_for_kbit_training,
)


# Config for loading the model in 4 bits
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # original is 32 bit
    bnb_4bit_quant_type="nf4",  # gaussian distribution
    bnb_4bit_use_double_quant=True,  # 32 -> 8 -> 4 bits
    bnb_4bit_compute_dtype=torch.float16,  # compute in float16
)

# Load the model with our config
base_model_id = "mistralai/Ministral-3-3B-Base-2512"
base_model = Mistral3ForConditionalGeneration.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,  # use our quantization config
    device_map="auto",  # use CUDA if available
)

# Load the tokenizer
tokenizer = MistralCommonBackend.from_pretrained(
    base_model_id
)  # tokenization specific to Mistral models

# Enable gradient checkpointing
base_model.gradient_checkpointing_enable()
# Quantization-aware training
base_model = prepare_model_for_kbit_training(base_model)
# Save the quantized base model
base_model.save_pretrained("Ministral-3-3B-Base-2512-bnb-nf4")
# Save the tokenizer
tokenizer.save_pretrained("Ministral-3-3B-Base-2512-bnb-nf4")
