def preprocess(examples):
    # tokenize the prompts
    model_inputs = tokenizer(
        examples["text"], 
        max_length=512, 
        truncation=True, 
        padding="max_length"
    )
    # tokenize the targets and shift them into labels
    labels = tokenizer(
        examples["text_target"], 
        max_length=512, 
        truncation=True, 
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_ds = train_ds.map(
    preprocess,
    batched=True,
    remove_columns=["text","text_target"]
)
test_ds = test_ds.map(
    preprocess,
    batched=True,
    remove_columns=["text","text_target"]
)
