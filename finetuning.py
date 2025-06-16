import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW
)
from peft import LoraConfig, get_peft_model, TaskType

class PairwiseRewardDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=256):
        self.examples = [json.loads(l) for l in open(jsonl_path)]
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        p, c, r = ex["prompt"], ex["chosen"], ex["rejected"]
        # tokenize prompt+response
        chosen = self.tok(p + c,
                          truncation=True,
                          padding="max_length",
                          max_length=self.max_length,
                          return_tensors="pt")
        rejected = self.tok(p + r,
                            truncation=True,
                            padding="max_length",
                            max_length=self.max_length,
                            return_tensors="pt")
        return {
            "input_ids_chosen":    chosen.input_ids[0],
            "attention_mask_chosen": chosen.attention_mask[0],
            "input_ids_rejected":   rejected.input_ids[0],
            "attention_mask_rejected": rejected.attention_mask[0],
        }

def pairwise_loss(score_chosen, score_rejected):
    return -torch.log(torch.sigmoid(score_chosen - score_rejected) + 1e-12).mean()

def train_with_lora(
    jsonl_path: str,
    base_model: str = "bert-base-uncased",
    output_dir: str = "./reward_model_lora",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = "cuda"
):
    # 1) Load tokenizer & base model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,             # single scalar output
    )

    # 2) Wrap with LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # sequence classification
        inference_mode=False,
        r=8,                         # rank
        lora_alpha=16,
        lora_dropout=0.05,
    )
    model = get_peft_model(model, peft_config)
    model.to(device)

    # 3) Data
    ds = PairwiseRewardDataset(jsonl_path, tokenizer)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # 4) Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for batch in dl:
            optimizer.zero_grad()
            for k,v in batch.items(): batch[k] = v.to(device)

            # forward chosen / rejected
            sc = model(
                input_ids=batch["input_ids_chosen"],
                attention_mask=batch["attention_mask_chosen"],
            ).logits.squeeze(-1)
            sr = model(
                input_ids=batch["input_ids_rejected"],
                attention_mask=batch["attention_mask_rejected"],
            ).logits.squeeze(-1)

            loss = pairwise_loss(sc, sr)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(dl)
        print(f"Epoch {epoch} — Pairwise Loss: {avg:.4f}")

    # 5) Save the LoRA adapters (and base config)
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Saved LoRA-adapted model to {output_dir}")

if __name__ == "__main__":
    train_with_lora(
        jsonl_path="reward_pairs.jsonl",
        base_model="bert-base-uncased",
        epochs=3,
        batch_size=8,    # you can go smaller
        lr=3e-4
    )