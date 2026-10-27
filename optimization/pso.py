import numpy as np


def run_pso(
    objective,
    bounds,
    num_particles,
    num_iterations,
    inertia,
    cognitive,
    social,
    seed,
):
    # Reset NumPy so PSO and DE receive the same random start for the same seed
    np.random.seed(seed)

    # Convert the bounds into arrays to clip all coordinates at once
    lower_bounds = np.array([bound[0] for bound in bounds])
    upper_bounds = np.array([bound[1] for bound in bounds])
    dimension = len(bounds)

    # Initialize every particle uniformly inside the search space
    particles = np.zeros((num_particles, dimension))
    for i in range(dimension):
        particles[:, i] = np.random.uniform(
            lower_bounds[i], upper_bounds[i], num_particles
        )
    velocities = np.zeros((num_particles, dimension))

    # Start personal bests at infinity and let the first evaluation fill them
    personal_best = particles.copy()
    personal_best_scores = np.full(num_particles, np.inf)
    global_best = particles[0].copy()
    global_best_score = np.inf

    # Keep the best fitness of every iteration for the convergence curve
    history = []

    for iteration in range(num_iterations):
        # Evaluate each particle with the objective function
        for i in range(num_particles):
            score = objective(particles[i])

            # Save the best position found by this particle
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best[i] = particles[i].copy()

            # Save the best position found by the whole swarm
            if score < global_best_score:
                global_best_score = score
                global_best = particles[i].copy()

        # Record the best-so-far fitness before updating the velocities
        history.append(global_best_score)

        # Move particles using inertia, cognitive and social components
        for i in range(num_particles):
            r1 = np.random.rand()
            r2 = np.random.rand()
            velocities[i] = (
                inertia * velocities[i]
                + cognitive * r1 * (personal_best[i] - particles[i])
                + social * r2 * (global_best - particles[i])
            )
            particles[i] += velocities[i]

            # Clip positions so particles stay inside the search limits
            particles[i] = np.clip(particles[i], lower_bounds, upper_bounds)

    return global_best, global_best_score, history
