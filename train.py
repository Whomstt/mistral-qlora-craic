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
import mlflow


# Load the base quantized model
base_model = Mistral3ForConditionalGeneration.from_pretrained(
    "ministral-3-3b-base-4bit",
    device_map="auto",  # use CUDA if available
)

# Load the tokenizer
tokenizer = MistralCommonBackend.from_pretrained("ministral-3-3b-base-4bit")

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
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.1,
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
    report_to="mlflow",
)

# Model Training
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Track as MLflow Experiment
mlflow.set_experiment("mistral-qlora-craic")
mlflow.transformers.autolog()  # Enable autologging
with mlflow.start_run():
    # Log the configs
    mlflow.log_params(training_args.to_dict())
    mlflow.log_params(lora_config.to_dict())
    # Train the adapter
    trainer.train()
    # Save and log
    peft_model.save_pretrained("mistral-qlora-craic")
    tokenizer.save_pretrained("mistral-qlora-craic")
    mlflow.log_artifacts("mistral-qlora-craic")
