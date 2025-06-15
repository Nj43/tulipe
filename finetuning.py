# Phase 3.1 & 3.2: Dataset conversion and Reward Model Training

import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import torch
from datasets import Dataset, DatasetDict, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from trl import RewardTrainer, PairwiseDataCollator, create_reference_model_and_tokenizer

# --- Phase 3.1: Prepare pairwise preference dataset ---

INPUT_FILE = "grader_dataset.jsonl"  # your raw JSONL
OUTPUT_DIR = Path("processed_data")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_raw(path: str) -> List[Dict]:
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def to_pairwise(examples: List[Dict]) -> List[Dict]:
    pairs = []
    for ex in examples:
        prompt = ex["prompt"].strip()
        chosen = ex["c_response"].strip()
        rejected = ex["r_response"].strip()
        # normalize finish bracket
        # ensure single Action: Finish[...] at end:
        chosen = chosen.split("Action:")[0].strip() + "\nAction:" + chosen.split("Action:")[-1]
        rejected = rejected.split("Action:")[0].strip() + "\nAction:" + rejected.split("Action:")[-1]
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "idx": ex["idx"]})
    return pairs

raw = load_raw(INPUT_FILE)
pairs = to_pairwise(raw)

# train/val/test split by idx
idxs = list({p["idx"] for p in pairs})
random.shuffle(idxs)
n = len(idxs)
train_ids = set(idxs[:int(n*0.8)])
val_ids   = set(idxs[int(n*0.8):int(n*0.9)])
test_ids  = set(idxs[int(n*0.9):])

splits = {"train": [], "validation": [], "test": []}
for p in pairs:
    if p["idx"] in train_ids:
        splits["train"].append(p)
    elif p["idx"] in val_ids:
        splits["validation"].append(p)
    else:
        splits["test"].append(p)

ds = DatasetDict({k: Dataset.from_list(v) for k,v in splits.items()})
for split in ds:
    ds[split].to_json(OUTPUT_DIR / f"{split}.jsonl")

print("Dataset splits saved to", OUTPUT_DIR)

# --- Phase 3.2: Fine-tune a reward model on pairwise data ---

MODEL_NAME = "huggyllama/llama-7b"  # or your chosen base
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
# create model with scalar head
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=1
)

# Preprocessing
def tokenize_pair(ex):
    # we'll concatenate prompt + response
    chosen_text = ex["prompt"] + "\n" + ex["chosen"]
    rejected_text = ex["prompt"] + "\n" + ex["rejected"]
    c = tokenizer(chosen_text, truncation=True, max_length=512)
    r = tokenizer(rejected_text, truncation=True, max_length=512)
    return {"input_ids_chosen": c["input_ids"], "attention_mask_chosen": c["attention_mask"],
            "input_ids_rejected": r["input_ids"], "attention_mask_rejected": r["attention_mask"]}

tokenized = ds.map(tokenize_pair, remove_columns=ds["train"].column_names)

# DataCollider for pairwise
data_collator = PairwiseDataCollator(tokenizer)

# TrainingArguments
training_args = TrainingArguments(
    output_dir="reward_model",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-5,
    num_train_epochs=3,
    logging_steps=50,
    save_total_limit=2,
    fp16=torch.cuda.is_available()
)

# Metric: accuracy of preferring chosen over rejected
metric = load_metric("accuracy")

def compute_metrics(eval_preds):
    scores_chosen, scores_rejected, _ = eval_preds
    preds = (scores_chosen > scores_rejected).long()
    labels = torch.ones_like(preds)
    acc = metric.compute(predictions=preds.cpu().numpy(), references=labels.cpu().numpy())["accuracy"]
    return {"accuracy": acc}


trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("reward_model")
