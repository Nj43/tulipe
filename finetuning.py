# sft_lora_fp16_full.py
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# ────────────────────────────────
# 1) Hyperparameters & paths
# ────────────────────────────────
RAW_JSON         = "data/train.json"            # your JSON file
OUTPUT_DIR       = "lora-adapter/llama7b-sft"   # where to save the LoRA adapter

MODEL_NAME       = "meta-llama/Llama-2-7b-hf"
MAX_LEN          = 512
TRAIN_EPOCHS     = 3
LR               = 3e-4
BATCH_PER_DEVICE = 4    # real forward batch size
GRAD_ACCUM       = 2    # to get effective batch size = BATCH_PER_DEVICE * GRAD_ACCUM

# ────────────────────────────────
# 2) Load & preprocess dataset
# ────────────────────────────────
print("▶ Loading raw JSON…")
raw = load_dataset("json", data_files=RAW_JSON)["train"]

print("▶ Loading tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
tokenizer.pad_token_id = tokenizer.eos_token_id

def tokenize_and_mask(example):
    prompt = example["prompt"]
    # append EOS so the model knows where to stop
    target = example["c_response"].strip() + tokenizer.eos_token

    # Concatenate and tokenize
    full = prompt + target
    tokens = tokenizer(
        full,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
    )

    # Build labels: ignore the prompt portion
    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    labels = [-100] * prompt_len + tokens["input_ids"][prompt_len:]
    labels += [-100] * (MAX_LEN - len(labels))

    return {
        "input_ids":      tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels":         labels,
    }

print("▶ Tokenizing & creating labels…")
tok = raw.map(
    tokenize_and_mask,
    remove_columns=raw.column_names,
    batched=False,
)

print("▶ Splitting into train/test…")
ds = tok.train_test_split(test_size=0.05, seed=42)

# ────────────────────────────────
# 3) Define collate_fn
# ────────────────────────────────
def collate_fn(batch):
    """
    batch: List[{"input_ids":List[int], "attention_mask":List[int], "labels":List[int]}]
    returns a dict of torch tensors
    """
    input_ids = torch.tensor([ex["input_ids"]      for ex in batch], dtype=torch.long)
    attention_mask = torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long)
    labels    = torch.tensor([ex["labels"]         for ex in batch], dtype=torch.long)
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }

# ────────────────────────────────
# 4) Load base model in FP16 and attach LoRA
# ────────────────────────────────
print("▶ Loading LLaMA-2-7B in FP16 on GPU 0…")
base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map={"": 0},       # pin all layers to cuda:0
    low_cpu_mem_usage=True,
)

print("▶ Attaching LoRA adapter…")
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(base, lora_cfg)

# Sanity check: ensure every parameter is on cuda
for name, param in model.named_parameters():
    if param.device.type != "cuda":
        raise RuntimeError(f"Parameter {name} is on {param.device}, expected cuda")

# ────────────────────────────────
# 5) Configure & run SFTTrainer
# ────────────────────────────────
print("▶ Configuring SFTTrainer…")
sft_config = SFTConfig(
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
    args=sft_config,
    data_collator=collate_fn,  # use our custom collator
)

print("▶ Starting training…")
trainer.train()

# ────────────────────────────────
# 6) Save LoRA adapter only
# ────────────────────────────────
print(f"▶ Saving LoRA adapter to {OUTPUT_DIR} …")
model.save_pretrained(OUTPUT_DIR)
print("✅ LoRA adapter saved. Done!")
