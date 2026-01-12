# Import the necessary libraries
import torch
from transformers import (
    BitsAndBytesConfig,
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
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

# Enable gradient checkpointing
base_model.gradient_checkpointing_enable()
# Quantization-aware training
base_model = prepare_model_for_kbit_training(base_model)
# Save the quantized base model
base_model.save_pretrained("ministral-3-3b-base-4bit")

# Setup for the tokenizer
tokenizer = MistralCommonBackend.from_pretrained(
    base_model_id
)  # tokenization specific to Mistral models

# Load the dataset
dataset = load_dataset("json", data_files="training_data.jsonl", split="train")


# Tokenize the dataset
def tokenize(sample):
    return tokenizer(
        sample["text"], truncation=True, padding="max_length", max_length=64
    )


tokenized_dataset = dataset.map(tokenize, batched=True)

# Lora Configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
peft_model = get_peft_model(base_model, lora_config)

# Training Configuration
training_args = TrainingArguments(
    output_dir="./qlora-checkpoint",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    report_to="none",
)

# Model Training
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()

# Save the trained LoRA
peft_model.save_pretrained("mistral-qlora-craic")
tokenizer.save_pretrained("mistral-qlora-craic")
