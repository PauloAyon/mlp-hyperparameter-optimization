from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred):
    # Compute binary scores for the positive malignant class
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )

    # Keep a fixed 2x2 layout: benign first, malignant second
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Return all values together so the tasks can report them
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": matrix,
    }
