#!/usr/bin/env python3
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
)
from trl import RewardTrainer, RewardConfig

def load_comparison_json(path):
    """Load your JSON comparison data, filter out any records
       missing either c_response or r_response."""
    raw = json.load(open(path, 'r', encoding='utf-8'))
    examples = []
    for rec in raw:
        prompt = rec.get("prompt", "").strip()
        chosen = rec.get("c_response", "").strip()
        rejected = rec.get("r_response", "").strip()
        # simple sanity check
        if prompt and chosen and rejected:
            examples.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })
    return examples

def make_dataset(examples, tokenizer, max_length=512):
    """Turn the list of dicts into a Hugging Face Dataset suitable for RewardTrainer.
       RewardTrainer expects columns ['input_ids', 'attention_mask', 'labels'] for both
       chosen and rejected, so we return a Dataset with two fields:
         - 'chosen_input_ids', 'chosen_attention_mask'
         - 'rejected_input_ids', 'rejected_attention_mask'
    """
    def tokenize_pair(ex):
        # concatenate prompt + response
        chosen_enc = tokenizer(
            ex["prompt"],
            ex["chosen"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        rejected_enc = tokenizer(
            ex["prompt"],
            ex["rejected"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
        }

    ds = Dataset.from_list(examples)
    ds = ds.map(tokenize_pair, remove_columns=ds.column_names)
    return ds

def main():
    data_path = "comparison_data.json"  # your JSON file
    output_dir = "reward_model"
    base_model = "distilroberta-base"   # or your preferred backbone

    # 1) Load data
    examples = load_comparison_json(data_path)
    print(f"> Loaded {len(examples)} comparison examples")

    # 2) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=1
    )

    # 3) Build HF Dataset
    ds = make_dataset(examples, tokenizer)
    # split
    split = ds.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"> Train size: {len(train_ds)}, Eval size: {len(eval_ds)}")

    # 4) Configure RewardTrainer
    reward_config = RewardConfig(
        # the name of the column with chosen vs rejected features:
        # by default, RewardTrainer looks for:
        #   - "chosen_input_ids", "chosen_attention_mask"
        #   - "rejected_input_ids", "rejected_attention_mask"
        reward_tokenizer=tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=200,
        save_total_limit=2,
        learning_rate=1e-5,
        fp16=torch.cuda.is_available(),
        logging_dir=f"{output_dir}/logs",
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        reward_config=reward_config,
    )

    # 5) Train!
    trainer.train()
    trainer.save_model(output_dir)
    print(f">>> Reward model saved to {output_dir}")

if __name__ == "__main__":
    main()
