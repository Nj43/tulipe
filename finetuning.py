from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split

# 1) Load your .json (expects a top-level list of dicts)
# If your file is simply `[ {...}, {...}, ... ]`, you can do:
raw = load_dataset("json", data_files="path/to/your/input.json", field=None)

# raw is a DatasetDict with a single split named "train"
# (even though it's the entire file)
print(raw)  
# DatasetDict({
#     train: Dataset({
#         features: [...],
#         num_rows: ...
#     })
# })

# 2) Split into train/validation/test (e.g. 80/10/10):
train_val, test = raw["train"].train_test_split(test_size=0.1, seed=42).values()
train, val   = train_val.train_test_split(test_size=0.1111, seed=42).values()
# 0.1111 of 90% ≈ 10% of original

data = DatasetDict({
    "train": train,
    "validation": val,
    "test": test,
})

# 3) Convert each example into a pairwise dict:
#    assuming each record has "c_response" and "r_response"
#    and you’ve already tokenized prompts+responses to match your reward model’s inputs.

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("your-base-model")

def make_pair(ex):
    # concatenate prompt + response, then tokenize
    chosen = tokenizer(ex["prompt"] + ex["c_response"], truncation=True)
    rejected = tokenizer(ex["prompt"] + ex["r_response"], truncation=True)

    return {
        "chosen_input_ids":      chosen["input_ids"],
        "chosen_attention_mask": chosen["attention_mask"],
        "rejected_input_ids":     rejected["input_ids"],
        "rejected_attention_mask":rejected["attention_mask"],
    }

data = data.map(make_pair, remove_columns=raw["train"].column_names)

# now `data["train"]`, `data["validation"]`, `data["test"]` each have the four
# fields your RewardTrainer will expect.

# Save or inspect
print(data)
