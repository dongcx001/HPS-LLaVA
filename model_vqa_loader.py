labels = [l.tolist() if isinstance(l, torch.Tensor) else l for l in labels]
labels_padded = tokenizer.pad(
    {"input_ids": labels},
    padding="max_length",
    max_length=512,
    return_tensors="pt"
)["input_ids"]