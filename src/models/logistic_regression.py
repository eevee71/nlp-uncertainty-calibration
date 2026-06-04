import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def run_logistic_regression(X_train_raw, X_test_raw, y_train) -> np.ndarray:
    """Trains TF-IDF + Logistic Regression and returns test probabilities of shape [N, 2]."""

    print("Training TF-IDF + Logistic Regression")

    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    return model.predict_proba(X_test)