import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForSequenceClassification


# MODEL 1: TF-IDF + Logistic Regression

def run_logistic_regression(X_train_raw, X_test_raw, y_train) -> np.ndarray:
    """Trains TF-IDF + Logistic Regression. Returns test probabilities of shape [N, 2]."""

    print("Training TF-IDF + Logistic Regression")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)


# MODEL 2: MLP on Text Embeddings

def run_mlp_embeddings(X_train_raw, X_test_raw, y_train) -> np.ndarray:
    """Trains MLP on MiniLM embeddings and returns test probabilities"""

    print("Extracting MiniLM Embeddings & Training MLP")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    X_train = embedder.encode(X_train_raw, show_progress_bar=False)
    X_test = embedder.encode(X_test_raw, show_progress_bar=False)

    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)


# MODEL 3: Recurrent Neural Network LSTM (PyTorch)

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        emb = self.embedding(x)
        output, _ = self.lstm(emb)  # shape: (batch, seq_len, hidden_dim * 2)
        pooled = torch.mean(output, dim=1)
        return torch.sigmoid(self.fc(pooled)).squeeze(-1)


def run_lstm(X_train_raw, X_test_raw, y_train, max_words=5000, seq_len=50) -> np.ndarray:
    """Trains PyTorch LSTM and returns test probabilities"""

    print("Training PyTorch LSTM")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    word_counts = {}
    for text in X_train_raw:
        clean_text = text.lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "")
        for word in clean_text.split():
            word_counts[word] = word_counts.get(word, 0) + 1

    vocab = {word: i + 2 for i, (word, _) in
             enumerate(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:max_words - 2])}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

    def tokenize_and_pad(texts):
        sequences = []
        for text in texts:
            clean_text = text.lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "")
            seq = [vocab.get(word, 1) for word in clean_text.split()][:seq_len]
            padded = [0] * (seq_len - len(seq)) + seq
            sequences.append(padded)
        return torch.tensor(sequences, dtype=torch.long)

    X_train_tensor = tokenize_and_pad(X_train_raw)
    X_test_tensor = tokenize_and_pad(X_test_raw)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = SimpleLSTM(vocab_size=max_words + 2, emb_dim=64, hidden_dim=64).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(6):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_tensor_dev = X_test_tensor.to(device)
        test_preds = model(X_test_tensor_dev).cpu().numpy()

    return np.vstack([1 - test_preds, test_preds]).T


# MODEL 4: Small Transformer (DistilBERT)

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