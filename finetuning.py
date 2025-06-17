import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

# 1. load tokenizer & base model
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct", trust_remote_code=True)
base = AutoModelForSequenceClassification.from_pretrained(
    "tiiuae/falcon-7b-instruct",
    num_labels=1,  # we’ll output a *single* scalar score
    torch_dtype=torch.float16,
).cuda()

# 2. wrap in LoRA
lora_cfg = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj","v_proj"],  # Falcon’s attention proj.
    lora_dropout=0.05, bias="none",
)
reward_model = get_peft_model(base, lora_cfg)
reward_model.print_trainable_parameters()

# 3. collate function: encode prompt+response, return two inputs
def collate_fn(batch):
    # each item: {"prompt":…, "chosen":…, "rejected":…}
    enc = tokenizer(
        [b["prompt"] + "\n" + b["chosen"]     for b in batch],
        [b["prompt"] + "\n" + b["rejected"]   for b in batch],
        padding="longest", truncation=True, return_tensors="pt",
    )
    # enc.input_ids is shape (2*B, L)
    # so we split:
    B = len(batch)
    input_ids = enc.input_ids.view(2, B, -1).transpose(0,1)    # (B, 2, L)
    attn_mask = enc.attention_mask.view(2, B, -1).transpose(0,1)
    return {
      "input_ids_chosen":     input_ids[:,0].cuda(),
      "attention_mask_chosen":attn_mask[:,0].cuda(),
      "input_ids_rejected":   input_ids[:,1].cuda(),
      "attention_mask_rejected":attn_mask[:,1].cuda(),
    }

# 4. define a custom Trainer to optimize pairwise logistic loss
class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # forward
        out_ch = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
        ).logits.view(-1)    # (B,)
        out_rj = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
        ).logits.view(-1)    # (B,)

        # pairwise logistic loss: −log σ(s₊ − s₋)
        diff = out_ch - out_rj
        loss = torch.nn.functional.binary_cross_entropy_with_logits(diff, torch.ones_like(diff))
        return (loss, out_ch) if return_outputs else loss

# 5. prepare data loader
train_loader = DataLoader(your_reward_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# 6. train
trainer = RewardTrainer(
    model=reward_model,
    args=TrainingArguments(
        output_dir="rm-lora",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        max_steps=2000,
        fp16=True,
        logging_steps=50,
        save_steps=500,
    ),
    train_dataset=your_reward_dataset,
    data_collator=collate_fn,
)
trainer.train()
reward_model.save_pretrained("rm-lora")
