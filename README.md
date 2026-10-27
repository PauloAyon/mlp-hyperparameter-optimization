# MLP Classification and Hyperparameter Optimization

This project classifies the cancer dataset in `data/cancer.csv` with a multilayer perceptron (MLP) from scikit-learn. It first evaluates a fixed baseline network, then tunes the network with particle swarm optimization (PSO) and differential evolution (DE), and finally compares both optimizers on the untouched test set.

The optimization code is written directly with NumPy. No external PSO/DE packages or deep learning frameworks are used; all models are scikit-learn MLPs.

## Tasks

1. **Baseline classification:** train an MLP with `(3, 2)` hidden neurons and `max_iter=1000`, then report test accuracy, precision, recall and the confusion matrix.
2. **Hyperparameter optimization:** tune the number of hidden layers, neurons per layer, `alpha`, `learning_rate_init` and activation function with PSO and DE. Both optimizers use the same bounds, evaluation budget and seeds.
3. **Comparison:** compare convergence curves and test metrics. The best model of each optimizer is retrained on train plus validation and evaluated once on the test set.

## Dataset

`data/cancer.csv` has no header, 699 rows and 11 numeric columns:

| Columns | Content |
| --- | --- |
| 0 to 8 | Nine input features |
| 9 | One-hot benign indicator (`1` = benign) |
| 10 | One-hot malignant indicator / label (`1` = malignant) |

The project uses column index `10` as the target, so the positive class for binary precision and recall is malignant. The data is split once into 60% training, 20% validation and 20% test, and the scaler is fitted only on the training split to avoid data leakage.

## Project structure

```
.
├── main.py                          # single entry point and experiment settings
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── cancer.csv                   # original dataset, kept unchanged
│   └── loader.py                    # loading, 60/20/20 split and scaling
├── optimization/
│   ├── pso.py                       # particle swarm optimization
│   ├── differential_evolution.py    # differential evolution
│   └── search_space.py              # bounds and hyperparameter decoding
├── processing/
│   ├── model.py                     # MLP builder, fitness and final training
│   ├── metrics.py                   # accuracy, precision, recall and matrix
│   └── plots.py                     # all figures saved to output/
├── tasks/
│   ├── baseline.py                  # Task 1
│   ├── tuning.py                    # Task 2
│   └── comparison.py                # Task 3
└── output/                          # generated figures
```

## Requirements

- Python 3.11 or newer (tested with Python 3.14)
- numpy
- matplotlib
- pandas
- scikit-learn

All runtime dependencies are listed in `requirements.txt`.

## Installation

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

## How to run

Run the only entry point from the project root:

```bash
python main.py
```

The complete experiment uses 3 runs per optimizer and usually takes a few minute. The figures are saved in `output/` and no interactive plot window is required.

## Configuration

The main settings are easy to change:

| Parameter | Location | Default |
| --- | --- | --- |
| Baseline architecture | `main.py` | `(3, 2)` |
| Baseline `max_iter` | `main.py` | `1000` |
| Search bounds | `optimization/search_space.py` | 1 to 3 layers, 10 to 150 neurons, `alpha` 0.0001 to 0.1, learning rate 0.0001 to 0.1, `relu` or `tanh` |
| PSO particles | `main.py` | 10 |
| PSO iterations | `main.py` | 12 |
| PSO inertia, cognitive and social coefficients | `main.py` | 1.2, 0.15, 1.9 |
| DE population | `main.py` | 10 |
| DE generations | `main.py` | 12 |
| DE mutation and crossover | `main.py` | 0.8, 0.7 |
| Optimization runs per method | `main.py` | 3 |
| Base random seed | `main.py` | 42 |
| Candidate training `max_iter` | `main.py` | 500 |
| Final training `max_iter` | `main.py` | 2000 |
| Fitness function | `processing/model.py` | `1 - validation accuracy` |

To perform a quick test, reduce the number of particles or individuals and the number of iterations or generations.

## Evaluation protocol

- The fitness minimized by both methods is `1 - validation accuracy`.
- Both methods receive the same hyperparameter bounds.
- Each PSO run uses `10 particles x 12 iterations = 120` model evaluations.
- Each DE run uses 120 evaluations as well. The initial population counts as generation 1, followed by 11 evolutionary generations, for `10 x 12 = 120` evaluations.
- PSO and DE use the same base seed plus the run index (`42`, `43`, `44`), and the MLP receives the same seed for each paired run.
- The baseline is trained on the 60% training split. The best run of each optimizer is retrained on train plus validation and evaluated once on the test set.
- Accuracy is reported together with binary precision and recall for the malignant class.

The 60/20/20 split is intentionally not seeded, so the reported values change between program executions. The optimizer seeds make the PSO and DE comparison fair within a run.

## Expected output

After execution, `output/` contains:

| File | Content |
| --- | --- |
| `baseline_confusion_matrix.png` | Test confusion matrix of the fixed `(3, 2)` baseline |
| `convergence_comparison.png` | Fitness minimization and validation accuracy for PSO and DE, with mean lines and min-max bands over the 3 runs |
| `test_confusion_matrices.png` | Test confusion matrices of the best PSO and DE models side by side |
| `metrics_comparison.png` | Grouped test accuracy, precision and recall bars for PSO and DE |

The terminal reports the split sizes, baseline metrics, best validation accuracy and selected hyperparameters for each optimizer, and the final test metrics.

A new execution can produce different values because the split is not seeded.
