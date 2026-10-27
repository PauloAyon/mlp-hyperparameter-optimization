from optimization.differential_evolution import run_differential_evolution
from optimization.pso import run_pso
from optimization.search_space import BOUNDS, decode_position
from processing.model import evaluate_position


def tune_with_pso(X_train, y_train, X_val, y_val, config, seed, max_iter):
    runs = []

    # Repeat PSO with the same seed sequence used by differential evolution
    for run_index in range(config["runs"]):
        run_seed = seed + run_index

        # Wrap the validation objective so PSO receives a one-argument function
        def objective(position):
            return evaluate_position(
                position, X_train, y_train, X_val, y_val, run_seed, max_iter
            )

        # Run PSO over the shared hyperparameter bounds
        best_position, best_score, history = run_pso(
            objective,
            BOUNDS,
            config["particles"],
            config["iterations"],
            config["inertia"],
            config["cognitive"],
            config["social"],
            run_seed,
        )

        # Store the best result and convergence history of this run
        runs.append(
            {
                "position": best_position,
                "fitness": best_score,
                "accuracy": 1.0 - best_score,
                "history": history,
            }
        )

    # Keep the run with the lowest validation fitness as the PSO result
    best_run = min(runs, key=lambda run: run["fitness"])

    # Decode the best position and return everything needed for reporting
    return {
        "runs": runs,
        "best_position": best_run["position"],
        "best_fitness": best_run["fitness"],
        "best_accuracy": best_run["accuracy"],
        "settings": decode_position(best_run["position"]),
    }


def tune_with_de(X_train, y_train, X_val, y_val, config, seed, max_iter):
    runs = []

    # Repeat DE with the same seed sequence used by PSO
    for run_index in range(config["runs"]):
        run_seed = seed + run_index

        # Wrap the validation objective so DE receives a one-argument function
        def objective(position):
            return evaluate_position(
                position, X_train, y_train, X_val, y_val, run_seed, max_iter
            )

        # Run DE over the same hyperparameter bounds
        best_position, best_score, history = run_differential_evolution(
            objective,
            BOUNDS,
            config["population"],
            config["generations"],
            config["mutation"],
            config["crossover"],
            run_seed,
        )

        # Store the best result and convergence history of this run
        runs.append(
            {
                "position": best_position,
                "fitness": best_score,
                "accuracy": 1.0 - best_score,
                "history": history,
            }
        )

    # Keep the run with the lowest validation fitness as the DE result
    best_run = min(runs, key=lambda run: run["fitness"])

    # Decode the best position and return everything needed for reporting
    return {
        "runs": runs,
        "best_position": best_run["position"],
        "best_fitness": best_run["fitness"],
        "best_accuracy": best_run["accuracy"],
        "settings": decode_position(best_run["position"]),
    }
