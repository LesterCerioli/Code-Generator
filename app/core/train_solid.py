import torch
from transformers import Trainer, TrainingArguments
from core.model import load_model
from core.tokenizer import load_tokenizer
from datasets import load_dataset

tokenizer = load_tokenizer()
model = load_model()

dataset = load_dataset("json", data_files="datasets/solid_theory.jsonl")

def tokenize(batch):
    return tokenizer(batch["prompt"], truncation=True, padding=True)

dataset = dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./solid_model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_steps=10,
    save_steps=500,
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

trainer.train()
