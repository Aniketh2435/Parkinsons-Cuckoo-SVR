"""Standard optimisation test functions, used to check that the Cuckoo Search operators work
before trusting them on the SVR problem. Every function has its global minimum f = 0."""
import numpy as np


def sphere(x):
    return float(np.sum(x ** 2))


def rastrigin(x):
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def rosenbrock(x):
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def ackley(x):
    n = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
                 - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)


# name -> (function, search domain [lo, hi] per dimension, position of the global minimum)
FUNCTIONS = {
    "Sphere": (sphere, (-100.0, 100.0), 0.0),
    "Rastrigin": (rastrigin, (-5.12, 5.12), 0.0),
    "Rosenbrock": (rosenbrock, (-5.0, 10.0), 1.0),
    "Ackley": (ackley, (-32.768, 32.768), 0.0),
}


class BenchmarkObjective:
    """Wraps a test function so it takes positions in [0, 1]^d, like the SVR objective.

    Keeps the same `log` / `n_evals` interface as `search_space.Objective`, so
    CuckooSearch and random_search run on it unchanged.
    """

    def __init__(self, name):
        self.name = name
        self.f, (self.lo, self.hi), self.x_opt = FUNCTIONS[name]
        self.log = []

    def __call__(self, x):
        fitness = self.f(self.lo + np.asarray(x) * (self.hi - self.lo))
        best = min(fitness, self.log[-1]["best_so_far"]) if self.log else fitness
        self.log.append({"eval": len(self.log) + 1, "fitness": fitness, "best_so_far": best})
        return fitness

    def evaluate_many(self, xs):
        return [self(x) for x in xs]

    @property
    def n_evals(self):
        return len(self.log)
