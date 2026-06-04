import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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