import numpy as np
from src.data_loader import DisasterDataLoader
from src.metrics import CalibrationAnalyzer
from src.models import run_logistic_regression, run_mlp_embeddings, run_lstm, run_transformer


def main():

    X_train, X_test, y_train, y_test = DisasterDataLoader('data/train.csv').load_splits()

    models = {
        "Logistic Regression TF-IDF": run_logistic_regression,
        "MLP Embeddings": run_mlp_embeddings,
        "LSTM Recurrent Net": run_lstm,
        "Transformer BERT-Tiny": run_transformer
    }

    all_model_probabilities = {}
    # generate test probabilities
    for name, func in models.items():
        print(f"Training and evaluating: {name}\n")

        probabilities = func(X_train, X_test, y_train)
        all_model_probabilities[name] = probabilities

    analyzer = CalibrationAnalyzer(n_bins=10)

    print("\nEvaluation\n")
    for name, probs in all_model_probabilities.items():
        predictions = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)

        analyzer.print_standard_metrics(y_test, predictions, model_name=name)
        analyzer.calculate_ece(y_test, confidences, predictions, model_name=name)
        analyzer.plot_reliability_diagram(y_test, confidences, predictions, model_name=name)
        analyzer.analyze_highly_confident_errors(X_test, y_test, confidences, predictions, model_name=name, top_k=3)


if __name__ == '__main__':
    main()