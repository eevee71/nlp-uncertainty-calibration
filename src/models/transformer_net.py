import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification


def run_transformer(X_train_raw, X_test_raw, y_train) -> np.ndarray:
    """Fine-tunes BERT-Tiny and returns test probabilities"""

    print("Fine-tuning BERT-Tiny")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "prajjwal1/bert-tiny"

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    train_encodings = tokenizer(list(X_train_raw), truncation=True, padding=True, max_length=64, return_tensors="pt")
    test_encodings = tokenizer(list(X_test_raw), truncation=True, padding=True, max_length=64, return_tensors="pt")

    y_train_t = torch.tensor(y_train, dtype=torch.long)

    dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], y_train_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(3):
        for b_ids, b_mask, b_labels in loader:
            b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        b_ids_test = test_encodings['input_ids'].to(device)
        b_mask_test = test_encodings['attention_mask'].to(device)
        test_outputs = model(input_ids=b_ids_test, attention_mask=b_mask_test)
        probs = torch.softmax(test_outputs.logits, dim=-1).cpu().numpy()

    return probs