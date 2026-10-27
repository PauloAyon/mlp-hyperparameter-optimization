import os
import warnings

from sklearn.exceptions import ConvergenceWarning

from data.loader import load_data
from tasks.baseline import run_baseline
from tasks.comparison import compare_optimizers
from tasks.tuning import tune_with_de, tune_with_pso

# Hide the expected warnings produced while many candidate networks are trained
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Baseline model settings
BASELINE_ARCHITECTURE = (3, 2)
BASELINE_MAX_ITER = 1000

# Shared seeds keep the PSO and DE comparison fair
SEED = 42
RUNS = 3

# PSO settings
PSO_CONFIG = {
    "particles": 10,
    "iterations": 12,
    "runs": RUNS,
    "inertia": 1.2,
    "cognitive": 0.15,
    "social": 1.9,
}

# Differential evolution settings
DE_CONFIG = {
    "population": 10,
    "generations": 12,
    "runs": RUNS,
    "mutation": 0.8,
    "crossover": 0.7,
}

# Use short runs for candidate models and a longer run for the final models
TUNING_MAX_ITER = 500
FINAL_MAX_ITER = 2000


def main():
    # Make sure the figure folder exists before any task runs
    os.makedirs("output", exist_ok=True)

    # Print the project title with an empty line for readability
    print("Multilayer Neural Networks and Hyperparameter Optimization")
    print()

    # Load the data and show the size of each split
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    print(f"  Train: {len(y_train)} | Validation: {len(y_val)} | Test: {len(y_test)}")

    # Run Task 1: train the fixed example architecture and test it
    print("\nRunning Task 1: Baseline Classification...")
    baseline = run_baseline(
        X_train,
        y_train,
        X_test,
        y_test,
        BASELINE_ARCHITECTURE,
        SEED,
        BASELINE_MAX_ITER,
    )
    print(f"  Architecture: {BASELINE_ARCHITECTURE}")
    print(f"    Accuracy: {baseline['accuracy']:.4f}")
    print(f"    Precision: {baseline['precision']:.4f}")
    print(f"    Recall: {baseline['recall']:.4f}")
    print("    Confusion matrix: output/baseline_confusion_matrix.png")

    # Run Task 2: optimize the MLP with PSO and differential evolution
    print("\nRunning Task 2: Hyperparameter Optimization...")
    print(
        f"  PSO: {PSO_CONFIG['particles']} particles, "
        f"{PSO_CONFIG['iterations']} iterations, {PSO_CONFIG['runs']} runs"
    )
    print(
        f"  Differential Evolution: population {DE_CONFIG['population']}, "
        f"{DE_CONFIG['generations']} generations, {DE_CONFIG['runs']} runs"
    )

    # Optimize with PSO and report its best validation result
    pso_results = tune_with_pso(
        X_train, y_train, X_val, y_val, PSO_CONFIG, SEED, TUNING_MAX_ITER
    )
    pso_settings = pso_results["settings"]
    print(
        f"    PSO best validation accuracy: "
        f"{pso_results['best_accuracy']:.4f}"
    )
    print(
        f"      Architecture: {pso_settings['hidden_layer_sizes']} | "
        f"alpha: {pso_settings['alpha']:.5f} | "
        f"lr: {pso_settings['learning_rate_init']:.5f} | "
        f"{pso_settings['activation']}"
    )

    # Optimize with DE and report its best validation result
    de_results = tune_with_de(
        X_train, y_train, X_val, y_val, DE_CONFIG, SEED, TUNING_MAX_ITER
    )
    de_settings = de_results["settings"]
    print(f"    DE  best validation accuracy: {de_results['best_accuracy']:.4f}")
    print(
        f"      Architecture: {de_settings['hidden_layer_sizes']} | "
        f"alpha: {de_settings['alpha']:.5f} | "
        f"lr: {de_settings['learning_rate_init']:.5f} | "
        f"{de_settings['activation']}"
    )

    # Run Task 3: retrain both winners and compare their test behavior
    print("\nRunning Task 3: PSO vs DE Comparison...")
    comparison = compare_optimizers(
        pso_results,
        de_results,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        SEED,
        FINAL_MAX_ITER,
    )
    pso_test = comparison["pso"]["metrics"]
    de_test = comparison["de"]["metrics"]
    print(
        f"  Test accuracy: PSO {pso_test['accuracy']:.4f} | "
        f"DE {de_test['accuracy']:.4f}"
    )
    print(
        f"  Test precision: PSO {pso_test['precision']:.4f} | "
        f"DE {de_test['precision']:.4f}"
    )
    print(
        f"  Test recall: PSO {pso_test['recall']:.4f} | "
        f"DE {de_test['recall']:.4f}"
    )
    print("  Figures saved in output/")


# Run the complete experiment when the file is executed directly
if __name__ == "__main__":
    main()
