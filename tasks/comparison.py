from processing.metrics import compute_metrics
from processing.model import fit_final_model
from processing.plots import (
    plot_convergence,
    plot_metrics_comparison,
    plot_test_confusion_matrices,
)


def compare_optimizers(
    pso_results,
    de_results,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    seed,
    max_iter,
):
    # Retrain the best PSO model on train plus validation and test it once
    pso_model, pso_settings = fit_final_model(
        pso_results["best_position"],
        X_train,
        y_train,
        X_val,
        y_val,
        seed,
        max_iter,
    )
    pso_predictions = pso_model.predict(X_test)
    pso_metrics = compute_metrics(y_test, pso_predictions)

    # Retrain the best DE model on train plus validation and test it once
    de_model, de_settings = fit_final_model(
        de_results["best_position"],
        X_train,
        y_train,
        X_val,
        y_val,
        seed,
        max_iter,
    )
    de_predictions = de_model.predict(X_test)
    de_metrics = compute_metrics(y_test, de_predictions)

    # Collect the best-so-far histories of every run for the convergence plot
    pso_histories = [run["history"] for run in pso_results["runs"]]
    de_histories = [run["history"] for run in de_results["runs"]]

    # Save the convergence, confusion matrix and grouped metric figures
    plot_convergence(pso_histories, de_histories)
    plot_test_confusion_matrices(
        pso_metrics["confusion_matrix"], de_metrics["confusion_matrix"]
    )
    plot_metrics_comparison(pso_metrics, de_metrics)

    # Return both test results and the decoded best hyperparameters
    return {
        "pso": {"metrics": pso_metrics, "settings": pso_settings},
        "de": {"metrics": de_metrics, "settings": de_settings},
    }
