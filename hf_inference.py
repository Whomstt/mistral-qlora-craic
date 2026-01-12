# Import necessary libraries
from peft import PeftModel
import torch
from transformers import (
    BitsAndBytesConfig,
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
)

# Load the base quantized model
base_model = Mistral3ForConditionalGeneration.from_pretrained(
    "ministral-3-3b-base-4bit",
    device_map="auto",  # use CUDA if available
)

# Load the trained LoRA
model = PeftModel.from_pretrained(base_model, "mistral-qlora-craic")
tokenizer = MistralCommonBackend.from_pretrained("mistral-qlora-craic")

# Take a prompt and generate text
while True:
    prompt = tokenizer(input("Enter your prompt: "), return_tensors="pt").to(
        model.device
    )
    output = model.generate(**prompt, max_new_tokens=30)
    print(
        tokenizer.decode(
            output[0][prompt["input_ids"].shape[-1] :], skip_special_tokens=True
        )
    )
