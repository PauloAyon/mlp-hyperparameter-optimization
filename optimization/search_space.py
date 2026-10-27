# Search limits for the five MLP hyperparameters
BOUNDS = [
    (1, 3.99),       # number of hidden layers
    (10, 150),       # neurons per hidden layer
    (0.0001, 0.1),   # L2 regularization strength
    (0.0001, 0.1),   # initial learning rate, capped for stable Adam training
    (0, 1.99),       # activation function index
]

# The encoded activation index selects one of these functions
ACTIVATIONS = ["relu", "tanh"]


def decode_position(position):
    # Convert the continuous layer and neuron values into integers
    num_layers = int(position[0])
    neurons = int(position[1])

    # Read the continuous regularization and learning rate values
    alpha = float(position[2])
    learning_rate = float(position[3])

    # Select the activation function from the encoded index
    activation = ACTIVATIONS[int(position[4])]

    # Repeat the same neuron count in every hidden layer
    hidden_layer_sizes = tuple([neurons] * num_layers)

    # Group the decoded values so the model builder can use them
    return {
        "hidden_layer_sizes": hidden_layer_sizes,
        "alpha": alpha,
        "learning_rate_init": learning_rate,
        "activation": activation,
    }
