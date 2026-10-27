from processing.metrics import compute_metrics
from processing.model import build_model
from processing.plots import plot_confusion_matrix


def run_baseline(X_train, y_train, X_test, y_test, architecture, seed, max_iter):
    # Use the example defaults for regularization, learning rate and activation
    settings = {
        "hidden_layer_sizes": architecture,
        "alpha": 0.0001,
        "learning_rate_init": 0.001,
        "activation": "relu",
    }

    # Build the fixed architecture and train it on the training split
    model = build_model(settings, seed, max_iter)
    model.fit(X_train, y_train)

    # Predict the test set once and compute the reported metrics
    predictions = model.predict(X_test)
    metrics = compute_metrics(y_test, predictions)

    # Save the baseline confusion matrix for the report
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        "Baseline MLP - Test Confusion Matrix",
        "baseline_confusion_matrix.png",
    )
    return metrics
