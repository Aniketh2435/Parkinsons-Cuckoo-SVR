"""Cuckoo Search (Yang & Deb, 2009), implemented from scratch for minimisation.

Biological inspiration: some cuckoo species lay eggs in other birds' nests. Each nest
holds one candidate solution (here: SVR hyperparameters + a feature subset). The algorithm uses:
  1. Levy flights      - new eggs are laid by long-tailed random jumps around current nests,
                         so most steps are small (exploitation) but some are very large
                         (exploration).
  2. Nest replacement  - a new egg replaces the nest it was laid from if it is fitter
                         (as in Yang & Deb's reference code; replacing a random nest lets one
                         good egg overwrite the whole population and the search stalls).
  3. Nest abandonment  - a fraction `pa` of eggs is discovered by the host bird and the
                         nest is rebuilt elsewhere (a biased random walk between nests).
  4. Elitism           - the best nest found so far is always kept.
"""
from math import gamma, pi, sin

import numpy as np


def levy_step(rng, size, beta=1.5):
    """Draw a Levy-distributed step with Mantegna's algorithm."""
    sigma_u = (gamma(1 + beta) * sin(pi * beta / 2)
               / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sigma_u, size)
    v = rng.normal(0, 1, size)
    return u / np.abs(v) ** (1 / beta)


class CuckooSearch:
    def __init__(self, objective, dim, n_nests=10, n_generations=10, pa=0.25,
                 alpha=1.0, beta=1.5, seed=42, warm_start=None, verbose=True):
        self.objective = objective
        self.dim = dim
        self.n_nests = n_nests
        self.n_generations = n_generations
        self.pa = pa              # probability that an egg is discovered (nest abandoned)
        self.alpha = alpha        # Levy step-size scale; the step is also proportional to the
                                  # distance from the best nest, so it shrinks as nests converge
        self.beta = beta          # Levy exponent, 1 < beta <= 2
        self.rng = np.random.default_rng(seed)
        self.warm_start = [] if warm_start is None else list(warm_start)   # known-good starting nests
        self.verbose = verbose
        self.history = []         # best fitness after each generation (convergence curve)
        self.mean_history = []    # mean fitness of the population after each generation

    def _lay_eggs_by_levy_flight(self, nests, fitness, best):
        """Operator 1 + 2: Levy flight from every nest, keep the egg if it beats its own nest."""
        eggs = []
        for i in range(self.n_nests):
            # Step size scales with distance from the best nest, so the swarm is
            # pulled towards good regions while each nest keeps its own position.
            step = self.alpha * levy_step(self.rng, self.dim, self.beta) * (nests[i] - best)
            eggs.append(np.clip(nests[i] + step * self.rng.standard_normal(self.dim), 0, 1))
        # Each egg depends only on its own nest and the best nest, so all eggs of a
        # generation can be scored in one parallel batch.
        for i, (egg, f_egg) in enumerate(zip(eggs, self.objective.evaluate_many(eggs))):
            if f_egg < fitness[i]:
                nests[i], fitness[i] = egg, f_egg

    def _abandon_nests(self, nests, fitness):
        """Operator 3: each dimension of each nest is discovered with probability pa;
        discovered nests are rebuilt by a random walk between two other nests."""
        discovered = self.rng.random((self.n_nests, self.dim)) < self.pa
        p1, p2 = self.rng.permutation(self.n_nests), self.rng.permutation(self.n_nests)
        step = self.rng.random((self.n_nests, 1)) * (nests[p1] - nests[p2])
        new_nests = np.clip(nests + step * discovered, 0, 1)
        rebuilt = np.flatnonzero(discovered.any(axis=1))
        for i, f_new in zip(rebuilt, self.objective.evaluate_many(new_nests[rebuilt])):
            if f_new < fitness[i]:
                nests[i], fitness[i] = new_nests[i], f_new

    def run(self):
        nests = self.rng.random((self.n_nests, self.dim))
        # Warm start: seed the first nests with given positions (e.g. the default model),
        # so the search starts from the baseline and can only improve on it.
        for k, x in enumerate(self.warm_start[:self.n_nests]):
            nests[k] = x
        fitness = np.array(self.objective.evaluate_many(nests))
        self.history.append(fitness.min())
        self.mean_history.append(fitness.mean())
        self._log(0, fitness)

        for gen in range(1, self.n_generations + 1):
            best = nests[fitness.argmin()].copy()     # Operator 4: elitism
            self._lay_eggs_by_levy_flight(nests, fitness, best)
            self._abandon_nests(nests, fitness)
            self.history.append(fitness.min())
            self.mean_history.append(fitness.mean())
            self._log(gen, fitness)

        i_best = fitness.argmin()
        return nests[i_best], fitness[i_best]

    def _log(self, gen, fitness):
        if self.verbose:
            print(f"  [CS] generation {gen:2d}/{self.n_generations}  best={fitness.min():.5f}  "
                  f"mean={fitness.mean():.5f}  evaluations={self.objective.n_evals}", flush=True)
