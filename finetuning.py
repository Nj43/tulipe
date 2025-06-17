# data_preprocess.py
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk

ds = load_dataset("json", data_files="data/train.json", field=None)

# We only want the “good” responses for SFT
def extract_fields(example):
    return {
      "input": example["prompt"],
      "target": example["c_response"]+"\n"  # include trailing newline for EOS
    }

ds = ds["train"].map(extract_fields, remove_columns=ds["train"].column_names)
ds = ds.train_test_split(test_size=0.05, seed=42)
ds["train"].save_to_disk("data/sft/train")
ds["test"].save_to_disk("data/sft/test")




# 1) Load tokeniser & model
model_name = "meta-llama/Llama-2-7b-hf"  # or your LLaMA7B HF path
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token_id = tokenizer.eos_token_id

base = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,           # bitsandbytes quant
    device_map="auto",
    quantization_config={"bnb_4bit_compute_dtype": torch.float16}
)

# 2) Attach LoRA
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],  # Llama-2 KV/QV projection modules
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(base, lora_cfg)

# 3) Load datasets
train_ds = load_from_disk("data/sft/train")
test_ds  = load_from_disk("data/sft/test")

# 4) SFTTrainer config
sft_config = SFTConfig(
    train_batch_size=8,
    micro_batch_size=4,
    num_train_epochs=3,
    learning_rate=3e-4,
    cutoff_len=512,              # max tokens per example
    val_set_size= len(test_ds),
    logging_steps=50,
    tensorboard_dir="runs/sft_lora",
)

# 5) Build the trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    args=sft_config,
)

# 6) Fine-tune
trainer.train()
# 7) Save just the LoRA adapter (small!)
model.save_pretrained("lora-adapter/llama7b-sft")

"""
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
base = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model = PeftModel.from_pretrained(base, "lora-adapter/llama7b-sft")

prompt = ds["test"][0]["input"]
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(out[0], skip_special_tokens=True))

"""