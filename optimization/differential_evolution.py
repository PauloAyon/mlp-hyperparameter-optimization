import numpy as np


def run_differential_evolution(
    objective,
    bounds,
    population_size,
    num_generations,
    mutation,
    crossover,
    seed,
):
    # Three different parents plus the target require at least four individuals
    if population_size < 4:
        raise ValueError("Differential evolution needs at least 4 individuals.")

    # Reset NumPy so PSO and DE receive the same random start for the same seed
    np.random.seed(seed)

    # Convert the bounds into arrays to clip every vector with one call
    lower_bounds = np.array([bound[0] for bound in bounds])
    upper_bounds = np.array([bound[1] for bound in bounds])
    dimension = len(bounds)

    # Create the initial population uniformly inside the search space
    population = np.zeros((population_size, dimension))
    for i in range(dimension):
        population[:, i] = np.random.uniform(
            lower_bounds[i], upper_bounds[i], population_size
        )

    # Evaluate the initial population and store the first best solution
    scores = np.zeros(population_size)
    for i in range(population_size):
        scores[i] = objective(population[i])
    best_index = int(np.argmin(scores))
    best_position = population[best_index].copy()
    best_score = float(scores[best_index])

    # The initial population counts as the first generation of the budget
    history = [best_score]

    # Evolve the population for the remaining generations
    for generation in range(1, num_generations):
        for i in range(population_size):
            # Pick three different parents that are not the current target
            candidates = [index for index in range(population_size) if index != i]
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

            # Build the mutant vector with the scaled difference of two parents
            mutant = population[r1] + mutation * (
                population[r2] - population[r3]
            )
            mutant = np.clip(mutant, lower_bounds, upper_bounds)

            # Apply binomial crossover and force at least one mutant gene
            trial = population[i].copy()
            gene = np.random.randint(dimension)
            for j in range(dimension):
                if np.random.rand() < crossover or j == gene:
                    trial[j] = mutant[j]

            # Evaluate the trial and keep it only if it improves the target
            trial_score = objective(trial)
            if trial_score < scores[i]:
                population[i] = trial
                scores[i] = trial_score

                # Update the best solution found so far
                if trial_score < best_score:
                    best_score = trial_score
                    best_position = trial.copy()

        # Store the best-so-far fitness after this generation
        history.append(best_score)

    return best_position, best_score, history
