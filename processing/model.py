import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from optimization.search_space import decode_position


def build_model(settings, seed, max_iter):
    # Create an MLP with the decoded hyperparameters and a fixed seed
    model = MLPClassifier(
        hidden_layer_sizes=settings["hidden_layer_sizes"],
        alpha=settings["alpha"],
        learning_rate_init=settings["learning_rate_init"],
        activation=settings["activation"],
        max_iter=max_iter,
        random_state=seed,
    )
    return model


def evaluate_position(position, X_train, y_train, X_val, y_val, seed, max_iter):
    # Decode the candidate position into MLP hyperparameters
    settings = decode_position(position)

    # Train the candidate only on the training split
    model = build_model(settings, seed, max_iter)
    model.fit(X_train, y_train)

    # Measure validation accuracy and return the fitness to minimize
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    return 1.0 - accuracy


def fit_final_model(position, X_train, y_train, X_val, y_val, seed, max_iter):
    # Decode the best hyperparameters found by the optimizer
    settings = decode_position(position)

    # Join train and validation data for the final training run
    X_train_val = np.vstack((X_train, X_val))
    y_train_val = np.concatenate((y_train, y_val))

    # Fit the final model with the complete training data
    model = build_model(settings, seed, max_iter)
    model.fit(X_train_val, y_train_val)
    return model, settings
