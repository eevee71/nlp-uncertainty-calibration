import numpy as np
from sklearn.neural_network import MLPClassifier
from sentence_transformers import SentenceTransformer


def run_mlp_embeddings(X_train_raw, X_test_raw, y_train) -> np.ndarray:
    """Trains MLP on MiniLM embeddings and returns test probabilities"""

    print("Extracting MiniLM Embeddings & Training MLP")

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    X_train = embedder.encode(list(X_train_raw), show_progress_bar=False)
    X_test = embedder.encode(list(X_test_raw), show_progress_bar=False)

    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
    model.fit(X_train, y_train)

    return model.predict_proba(X_test)