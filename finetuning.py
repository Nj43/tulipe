#!/usr/bin/env python
# train_reward_model_lora.py

import os
import json
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    default_data_collator,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from trl import RewardTrainer

def load_jsonl(path):
    """Load a .jsonl file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def build_dataset(data):
    """
    Expect each item in data to have keys:
      - prompt:      the instruction+context
      - c_response:  the preferred (chosen) model output
      - r_response:  the rejected output
    """
    records = {
        "prompt":   [item["prompt"]     for item in data],
        "chosen":   [item["c_response"] for item in data],
        "rejected": [item["r_response"] for item in data],
    }
    return Dataset.from_dict(records)

def tokenize_for_reward(examples, tokenizer, max_length=512):
    """
    Create the four fields that trl.RewardTrainer expects:
      input_ids_chosen, attention_mask_chosen,
      input_ids_rejected, attention_mask_rejected
    """
    chosen = tokenizer(
        examples["prompt"],
        examples["chosen"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    rejected = tokenizer(
        examples["prompt"],
        examples["rejected"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    return {
        "input_ids_chosen":   chosen["input_ids"],
        "attention_mask_chosen":   chosen["attention_mask"],
        "input_ids_rejected": rejected["input_ids"],
        "attention_mask_rejected": rejected["attention_mask"],
    }

def main():
    ###### 1. Paths & Hyperparams ######
    jsonl_path = "reward_data.jsonl"           # your comparison data
    output_dir = "reward_model_lora"           # where to save the LoRA-adapted reward model
    base_model  = "tiiuae/falcon-7b-instruct"  # reward‐backbone
    peft_r      = 8
    peft_alpha  = 16
    peft_dropout= 0.1

    train_batch_size = 4
    eval_batch_size  = 4
    num_train_epochs = 3
    logging_steps    = 50
    save_steps       = 500
    eval_steps       = 200
    max_length       = 512

    ###### 2. Load & Build Dataset ######
    raw = load_jsonl(jsonl_path)
    ds  = build_dataset(raw)
    # split 90/10
    ds = ds.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = ds["train"], ds["test"]

    ###### 3. Tokenizer & Model ######
    print("Loading tokenizer and model…")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # ensure tokenization pads to max_length
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_length

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,             # single scalar reward
        torch_dtype="auto",       # let HF pick float16 if available
        trust_remote_code=True,
    )

    ###### 4. Apply LoRA via PEFT ######
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=peft_r,
        lora_alpha=peft_alpha,
        lora_dropout=peft_dropout,
        target_modules=["q_proj", "v_proj"],  # Falcon‐style attention proj modules
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    ###### 5. Preprocess the data ######
    print("Tokenizing data…")
    train_ds = train_ds.map(
        lambda x: tokenize_for_reward(x, tokenizer, max_length),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    eval_ds = eval_ds.map(
        lambda x: tokenize_for_reward(x, tokenizer, max_length),
        batched=True,
        remove_columns=eval_ds.column_names,
    )

    ###### 6. Setup TrainingArguments ######
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=2,
        learning_rate=2e-5,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    ###### 7. RewardTrainer ######
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
    )

    ###### 8. Train! ######
    trainer.train()
    trainer.save_model(output_dir)

    print(f"\n✅ Finished. Your LoRA‐adapted reward model is in `{output_dir}`")

if __name__ == "__main__":
    main()
