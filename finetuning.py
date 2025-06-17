# train_reward_model_lora.py

import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import RewardTrainer, RewardConfig

# ──────────────────────────────────────────────────────────────────────────────
#                  1)  CONFIGURE PATHS & HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
DATA_JSON = "comparison_data.json"      # ← your JSON file
OUTPUT_DIR = "falcon7b_reward_lora"     # where to save adapters & tokenizer

MODEL_NAME = "tiiuae/falcon-7b-instruct"
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 3
LOGGING_STEPS = 50
SAVE_STEPS = 200

# LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]   # adjust if necessary


# ──────────────────────────────────────────────────────────────────────────────
#     2)  LOAD & PREP DATASET
# ──────────────────────────────────────────────────────────────────────────────
raw = json.load(open(DATA_JSON))   # list of dicts
records = []
for rec in raw:
    prompt  = rec["prompt"].strip()
    chosen  = rec["c_response"].strip()
    rejected= rec["r_response"].strip()
    records.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

ds = Dataset.from_list(records)
# simple 90/10 split
ds = ds.train_test_split(test_size=0.1)
train_ds = ds["train"]
eval_ds  = ds["test"]


# ──────────────────────────────────────────────────────────────────────────────
# 3)  TOKENIZER & BASE MODEL → reward head (1‐dim)
# ──────────────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,  padding_side="right")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    torch_dtype=torch.float16
)


# ──────────────────────────────────────────────────────────────────────────────
#               4)  WRAP WITH LoRA (peft)
# ──────────────────────────────────────────────────────────────────────────────
lora_cfg = LoraConfig(
    task_type       = TaskType.SEQ_CLS,
    inference_mode  = False,
    r               = LORA_R,
    lora_alpha      = LORA_ALPHA,
    lora_dropout    = LORA_DROPOUT,
    target_modules  = TARGET_MODULES,
)
model = get_peft_model(model, lora_cfg)


# ──────────────────────────────────────────────────────────────────────────────
# 5)  DEFINE TrainingArguments & RewardTrainer
# ──────────────────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir              = OUTPUT_DIR,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    gradient_accumulation_steps = 1,
    learning_rate           = LR,
    num_train_epochs        = EPOCHS,
    logging_steps           = LOGGING_STEPS,
    save_steps              = SAVE_STEPS,
    save_total_limit        = 2,
    fp16                    = True,
    evaluation_strategy     = "steps",
)

reward_config = RewardConfig(
    prompt_column   = "prompt",
    chosen_column   = "chosen",
    rejected_column = "rejected",
)

trainer = RewardTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_ds,
    eval_dataset    = eval_ds,
    tokenizer       = tokenizer,
    reward_config   = reward_config,
)


# ──────────────────────────────────────────────────────────────────────────────
#                        6)  TRAIN & SAVE
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    trainer.train()
    # save LoRA adapters + config
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Saved fine‐tuned reward model at {OUTPUT_DIR}")
