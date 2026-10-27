import os

import matplotlib

# Use a non-interactive backend so figures can be saved in any environment
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# All figures are written to this folder
OUTPUT_DIR = "output"


def save_figure(fig, filename):
    # Create the output folder if it does not exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save at a readable resolution and release the figure from memory
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def add_confusion_matrix(ax, matrix, title, vmax):
    # Draw the count matrix with a fixed scale for a fair comparison
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)

    # Write the count inside every cell with a readable text color
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > vmax / 2 else "black"
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color=color,
                fontsize=12,
            )

    # Add labels and a readable class order
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Malignant"])
    ax.set_yticklabels(["Benign", "Malignant"])
    return image


def plot_confusion_matrix(matrix, title, filename):
    # Convert the matrix to NumPy so its size and maximum are easy to read
    matrix = np.array(matrix)

    # Create a single confusion matrix figure for the baseline model
    fig, ax = plt.subplots(figsize=(5.4, 4.6), layout="constrained")
    image = add_confusion_matrix(ax, matrix, title, matrix.max())
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Samples")
    save_figure(fig, filename)


def plot_test_confusion_matrices(pso_matrix, de_matrix):
    # Convert both matrices to NumPy before computing the shared scale
    pso_matrix = np.array(pso_matrix)
    de_matrix = np.array(de_matrix)

    # Use the same color scale so both matrices can be compared directly
    vmax = max(pso_matrix.max(), de_matrix.max())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), layout="constrained")

    # Draw the PSO and DE test matrices side by side
    image = add_confusion_matrix(axes[0], pso_matrix, "PSO", vmax)
    add_confusion_matrix(axes[1], de_matrix, "Differential Evolution", vmax)

    # Share one colorbar between the two panels
    fig.colorbar(image, ax=axes.tolist(), fraction=0.035, pad=0.04, label="Samples")
    fig.suptitle("Test Confusion Matrices")
    save_figure(fig, "test_confusion_matrices.png")


def plot_convergence(pso_histories, de_histories):
    # Convert the run histories into arrays for mean and min-max bands
    pso_fitness = np.array(pso_histories, dtype=float)
    de_fitness = np.array(de_histories, dtype=float)

    # Validate that both methods have the same number of rounds
    if pso_fitness.shape[1] != de_fitness.shape[1]:
        raise ValueError("PSO and DE histories must have the same length.")

    # Build the generation axis shared by both panels
    generations = np.arange(1, pso_fitness.shape[1] + 1)

    # Keep the optimizer colors and markers consistent in both panels
    series = [
        (pso_fitness, "PSO", "tab:blue", "o"),
        (de_fitness, "DE", "tab:orange", "s"),
    ]

    # Create one figure with fitness and validation accuracy panels
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")

    # Show the mean best-so-far fitness and the run-to-run range
    for values, label, color, marker in series:
        mean = values.mean(axis=0)
        low = values.min(axis=0)
        high = values.max(axis=0)
        axes[0].plot(
            generations, mean, marker=marker, color=color, linewidth=2, label=label
        )
        axes[0].fill_between(generations, low, high, color=color, alpha=0.2)

    # Label the minimization panel
    axes[0].set_title("Fitness Minimization")
    axes[0].set_xlabel("Iteration / generation")
    axes[0].set_ylabel("Fitness (1 - validation accuracy)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Convert the same histories into validation accuracy
    for values, label, color, marker in series:
        accuracy = 1.0 - values
        mean = accuracy.mean(axis=0)
        low = accuracy.min(axis=0)
        high = accuracy.max(axis=0)
        axes[1].plot(
            generations, mean, marker=marker, color=color, linewidth=2, label=label
        )
        axes[1].fill_between(generations, low, high, color=color, alpha=0.2)

    # Label the accuracy panel
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Iteration / generation")
    axes[1].set_ylabel("Validation accuracy")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    # Add a common title and save the figure
    fig.suptitle("PSO vs Differential Evolution Convergence")
    save_figure(fig, "convergence_comparison.png")


def plot_metrics_comparison(pso_metrics, de_metrics):
    # Collect the three test metrics for both optimizers
    metric_names = ["Accuracy", "Precision", "Recall"]
    pso_values = [
        pso_metrics["accuracy"],
        pso_metrics["precision"],
        pso_metrics["recall"],
    ]
    de_values = [
        de_metrics["accuracy"],
        de_metrics["precision"],
        de_metrics["recall"],
    ]

    # Place the two bars of each metric side by side
    positions = np.arange(len(metric_names))
    width = 0.35

    # Draw the grouped bars with the same colors used in the convergence plot
    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    pso_bars = ax.bar(
        positions - width / 2,
        pso_values,
        width,
        label="PSO",
        color="tab:blue",
    )
    de_bars = ax.bar(
        positions + width / 2,
        de_values,
        width,
        label="DE",
        color="tab:orange",
    )

    # Write each score above its bar for a precise reading
    for bars in (pso_bars, de_bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.004,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Format the axes and zoom near the observed values without hiding zero
    ax.set_xticks(positions)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Score")
    lower = max(0.0, min(pso_values + de_values) - 0.05)
    ax.set_ylim(lower, 1.02)
    ax.set_title("Test Metrics Comparison")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    # Save the final metrics figure
    save_figure(fig, "metrics_comparison.png")
