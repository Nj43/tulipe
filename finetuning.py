# sft_lora_train_fp16.py
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# ────────────────────────────────
# 1) Hyperparameters & paths
# ────────────────────────────────
RAW_JSON         = "data/train.json"
CACHE_DIR        = "data/sft_cache"
OUTPUT_DIR       = "lora-adapter/llama7b-sft-fp16"

MODEL_NAME       = "meta-llama/Llama-2-7b-hf"
MAX_LEN          = 512
TRAIN_EPOCHS     = 3
LR               = 3e-4
BATCH_PER_DEVICE = 4
GRAD_ACCUM       = 2

# ────────────────────────────────
# 2) Load & preprocess dataset
# ────────────────────────────────
raw = load_dataset("json", data_files=RAW_JSON)["train"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
tokenizer.pad_token_id = tokenizer.eos_token_id

def tokenize_and_mask(example):
    prompt = example["prompt"]
    target = example["c_response"] + tokenizer.eos_token

    full   = prompt + target
    tokens = tokenizer(
        full,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
    )

    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    labels     = [-100]*prompt_len + tokens["input_ids"][prompt_len:]
    labels    += [-100]*(MAX_LEN - len(labels))

    return {
        "input_ids":      tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels":         labels,
    }

tok = raw.map(
    tokenize_and_mask,
    remove_columns=raw.column_names,
    batched=False,
)

ds = tok.train_test_split(test_size=0.05, seed=42)
os.makedirs(CACHE_DIR, exist_ok=True)
ds["train"].save_to_disk(f"{CACHE_DIR}/train")
ds["test"].save_to_disk(f"{CACHE_DIR}/test")


# ────────────────────────────────
# 3) Load base model in FP16 + pin to GPU
# ────────────────────────────────
base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map={"": 0},       # pin all to cuda:0 (or use "auto" if you prefer)
    low_cpu_mem_usage=True,
)

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(base, lora_cfg)

# sanity check
for n, p in model.named_parameters():
    if p.device.type != "cuda":
        raise RuntimeError(f"Param {n} on {p.device}")

# ────────────────────────────────
# 4) SFTTrainer config & launch
# ────────────────────────────────
config = SFTConfig(
    per_device_train_batch_size=BATCH_PER_DEVICE,
    gradient_accumulation_steps=GRAD_ACCUM,
    per_device_eval_batch_size=BATCH_PER_DEVICE,

    num_train_epochs=TRAIN_EPOCHS,
    learning_rate=LR,
    cutoff_len=MAX_LEN,

    logging_steps=50,
    logging_dir="runs/sft_lora_fp16",
    save_strategy="steps",
    save_steps=500,
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    do_eval=True,
    eval_steps=500,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    args=config,
    pack_sequences=False,
)

trainer.train()

# ────────────────────────────────
# 5) Save only the LoRA adapter
# ────────────────────────────────
model.save_pretrained(OUTPUT_DIR)
print(f"✅ LoRA adapter saved to {OUTPUT_DIR}")
