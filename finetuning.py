import json

def prepare_pairwise_jsonl(input_json_path: str, output_jsonl_path: str):
    """
    Reads a JSON list where each item has fields:
      - "prompt"
      - "c_response" (chosen/better)
      - "r_response" (rejected/worse)
    Writes out a JSONL file with one record per line:
      {"prompt": "...", "chosen": "...", "rejected": "..."}
    """
    data = json.load(open(input_json_path, 'r'))
    with open(output_jsonl_path, 'w') as fout:
        for item in data:
            record = {
                "prompt": item["prompt"].strip(),
                "chosen": item["c_response"].strip(),
                "rejected": item["r_response"].strip()
            }
            fout.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    prepare_pairwise_jsonl(
        input_json_path="grader_dataset.json",
        output_jsonl_path="reward_pairs.jsonl"
    )
    print("Wrote reward_pairs.jsonl")
    
    
    
import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW
)

class PairwiseRewardDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 256):
        self.examples = [json.loads(line) for line in open(jsonl_path)]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # concatenate prompt + response for each
        prompt = ex["prompt"]
        chosen = self.tokenizer(
            prompt + ex["chosen"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        rejected = self.tokenizer(
            prompt + ex["rejected"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids_chosen": chosen.input_ids.squeeze(0),
            "attention_mask_chosen": chosen.attention_mask.squeeze(0),
            "input_ids_rejected": rejected.input_ids.squeeze(0),
            "attention_mask_rejected": rejected.attention_mask.squeeze(0),
        }

def pairwise_loss(score_chosen, score_rejected):
    # Negative log-sigmoid of the difference
    diff = score_chosen - score_rejected
    return -torch.log(torch.sigmoid(diff) + 1e-12).mean()

def train_reward_model(
    jsonl_path: str,
    model_name: str = "bert-base-uncased",
    output_dir: str = "./reward_model",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    # 1) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,  # single scalar reward
    ).to(device)

    # 2) Prepare data
    dataset = PairwiseRewardDataset(jsonl_path, tokenizer)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3) Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            # move data to device
            for k, v in batch.items():
                batch[k] = v.to(device)

            # forward pass for chosen & rejected
            logits_chosen = model(
                input_ids=batch["input_ids_chosen"],
                attention_mask=batch["attention_mask_chosen"]
            ).logits.squeeze(-1)
            logits_rejected = model(
                input_ids=batch["input_ids_rejected"],
                attention_mask=batch["attention_mask_rejected"]
            ).logits.squeeze(-1)

            # compute pairwise loss
            loss = pairwise_loss(logits_chosen, logits_rejected)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch:>2} — Avg Pairwise Loss: {avg_loss:.4f}")

    # 4) Save
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved reward model to {output_dir}")

if __name__ == "__main__":
    train_reward_model(
        jsonl_path="reward_pairs.jsonl",
        model_name="bert-base-uncased",
        epochs=3,
        batch_size=16
    )