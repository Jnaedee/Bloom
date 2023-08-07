from datasets import load_dataset
from transformers import BloomTokenizerFast, BloomForTokenClassification, DataCollatorForTokenClassification, \
    AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset
import torch
import os

dataset = load_dataset("Impl.py", name="Kayan Dataset")

model_name = "bloom-560m"
tokenizer = BloomTokenizerFast.from_pretrained(f"bigscience/{model_name}", add_prefix_space=True)
model = BloomForTokenClassification.from_pretrained(f"bigscience/{model_name}")


def tokenizeInputs(inputs):
    tokenized_inputs = tokenizer(inputs["tokens"], truncation=True, max_length=512, is_split_into_words=True)
    word_ids = tokenized_inputs.word_ids()
    ner_tags = inputs["ner_tags"]
    labels = [ner_tags[word_id] for word_id in word_ids]
    tokenized_inputs["labels"] = labels

    return tokenized_inputs


tokenized_datasets = dataset.map(tokenizeInputs)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

model = AutoModelForTokenClassification.from_pretrained(f"bigscience/{model_name}", num_labels=12)
training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="epoch",  # Disabled for runtime evaluation
    evaluation_strategy="steps",  # "steps", # Disabled for runtime evaluation
    eval_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    num_train_epochs=2,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
