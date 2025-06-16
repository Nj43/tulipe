import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedModel,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
)
from typing import List, Dict

# 1) A tiny reward head on top of LLaMA-2-7B-Chat
class RewardHeadConfig(PretrainedConfig):
    model_type = "reward_head"
    def __init__(self, base_model_name: str = "meta-llama/Llama-2-7b-chat-hf", **kwargs):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name

class RewardHeadModel(PreTrainedModel):
    config_class = RewardHeadConfig

    def __init__(self, config: RewardHeadConfig):
        super().__init__(config)
        # load LLaMA as a causal LM
        base_cfg = AutoConfig.from_pretrained(config.base_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name, config=base_cfg, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        hidden_size = base_cfg.hidden_size
        # a little MLP to score the pooled hidden state
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        **kwargs,
    ) -> torch.FloatTensor:
        # get last hidden states from the causal LM
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # mean‐pool the last hidden state
        last_h = outputs.hidden_states[-1]  # (B, L, H)
        masked = last_h * attention_mask.unsqueeze(-1)
        pooled = masked.sum(1) / attention_mask.sum(1, keepdim=True)
        # project to a scalar
        reward = self.score(pooled).squeeze(-1)  # (B,)
        return reward

# 2) A Dataset that returns dicts with chosen/rejected fields
class PairwiseDataset(Dataset):
    def __init__(self, path: str, tokenizer):
        import json
        data = json.load(open(path))
        self.examples = data  # list of dict with prompt, chosen, rejected texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[i]
        # assume each has ex["chosen_text"], ex["rejected_text"]
        c = self.tokenizer(
            ex["chosen_text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )
        r = self.tokenizer(
            ex["rejected_text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "chosen_input_ids":   c.input_ids.squeeze(0),
            "chosen_attention_mask": c.attention_mask.squeeze(0),
            "rejected_input_ids":   r.input_ids.squeeze(0),
            "rejected_attention_mask": r.attention_mask.squeeze(0),
        }

# 3) A simple collator stacks them
def collate_fn(features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch = {}
    for k in features[0]:
        batch[k] = torch.stack([f[k] for f in features])
    return batch

# 4) Custom Trainer with margin ranking loss
class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # inputs: dict of four (B, L) tensors
        chosen_r = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
        )
        rejected_r = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
        )
        # we want chosen_r > rejected_r by margin=1.0
        target = torch.ones_like(chosen_r)
        loss_fct = nn.MarginRankingLoss(margin=1.0)
        loss = loss_fct(chosen_r, rejected_r, target)
        return (loss, (chosen_r, rejected_r)) if return_outputs else loss

# === Main ===
if __name__ == "__main__":
    from transformers import LlamaTokenizer

    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # load your JSON pairwise data
    train_ds = PairwiseDataset("phase3.1_train.json", tokenizer)
    eval_ds  = PairwiseDataset("phase3.1_valid.json", tokenizer)

    config = RewardHeadConfig(base_model_name=model_name)
    model = RewardHeadModel(config).to("cuda")

    training_args = TrainingArguments(
        output_dir="llama-reward-model",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-5,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        logging_steps=100,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
    )

    trainer.train()
