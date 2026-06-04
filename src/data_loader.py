import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Any


class DisasterDataLoader:
    """Handles dataset loading and splitting into clean raw text splits."""

    def __init__(self, filepath: str = 'data/train.csv'):
        self.filepath = filepath

    def load_splits(self) -> Tuple[Any, Any, Any, Any]:
        """Loads data and returns raw text splits (strings) and labels."""

        df = pd.read_csv(self.filepath)
        df = df.dropna(subset=['text', 'target'])
        X = df['text'].values
        y = df['target'].values

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        return X_train_raw, X_test_raw, y_train, y_test