"""Random Search: the non-bio-inspired reference optimiser.

Given the same number of fitness evaluations as Cuckoo Search, it shows whether the
Levy flights and nest abandonment actually beat sampling positions blindly.
"""
import numpy as np


def random_search(objective, dim, n_evals, seed=42, verbose=True, batch=25):
    rng = np.random.default_rng(seed)
    best_x, best_f = None, float("inf")
    done = 0
    while done < n_evals:
        xs = rng.random((min(batch, n_evals - done), dim))   # same samples as one at a time
        for x, f in zip(xs, objective.evaluate_many(xs)):
            if f < best_f:
                best_x, best_f = x, f
        done += len(xs)
        if verbose:
            print(f"  [RS] evaluation {done:3d}/{n_evals}  best={best_f:.5f}", flush=True)
    return best_x, best_f
