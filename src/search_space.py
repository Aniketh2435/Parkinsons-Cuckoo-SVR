"""Encoding of a nest: SVR hyperparameters + feature subset in one position vector.

A position x lies in [0, 1]^(3 + n_features):
  x[0:3]  -> C, gamma, epsilon, each on a log scale
  x[3:]   -> one value per feature; the feature is selected when its value > 0.5
             (threshold transfer from the continuous search space to a binary mask)
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

from .model import build_svr

HYPERPARAMS = [            # (name, low, high), all searched on a log10 scale
    ("C", 1e-1, 1e3),       # regularisation strength
    ("gamma", 1e-4, 1e1),   # RBF kernel width
    ("epsilon", 1e-3, 1.0), # width of the no-penalty tube (in standardised target units)
]
N_HYPER = len(HYPERPARAMS)
THRESHOLD = 0.5


def dim(n_features):
    return N_HYPER + n_features


def decode(x):
    """Position -> (hyperparameter dict, boolean feature mask)."""
    x = np.clip(x, 0, 1)
    params = {name: float(10 ** (np.log10(lo) + xi * np.log10(hi / lo)))
              for xi, (name, lo, hi) in zip(x[:N_HYPER], HYPERPARAMS)}
    mask = x[N_HYPER:] > THRESHOLD
    if not mask.any():                       # an empty feature set is invalid:
        mask[np.argmax(x[N_HYPER:])] = True  # keep the single most "selected" feature
    return params, mask


def encode(params, mask):
    """(hyperparameters, mask) -> position. Used to warm-start a nest at the default SVR."""
    hyper = [(np.log10(params[name]) - np.log10(lo)) / np.log10(hi / lo)
             for name, lo, hi in HYPERPARAMS]
    return np.concatenate([np.clip(hyper, 0, 1), np.where(mask, 0.75, 0.25)])


def default_position(n_features):
    """The default SVR (C=1, gamma='scale' = 1/n_features on scaled data, epsilon=0.1), all features."""
    return encode({"C": 1.0, "gamma": 1.0 / n_features, "epsilon": 0.1}, np.ones(n_features, bool))


def _fold_rmse(params, X, y, train_idx, val_idx):
    """RMSE of one SVR fitted on one CV training fold and scored on its validation fold."""
    model = build_svr(**params).fit(X[train_idx], y[train_idx])
    return root_mean_squared_error(y[val_idx], model.predict(X[val_idx]))


class Objective:
    """Fitness = mean k-fold cross-validated RMSE on the training set (lower is better).

    The test set is never touched during the search. Every call is logged for the
    convergence plots and the search-history CSV.
    """

    def __init__(self, X, y, feature_names, folds=3, seed=42):
        self.X, self.y, self.names = X, y, feature_names
        self.cv = KFold(folds, shuffle=True, random_state=seed)
        self.log = []

    def __call__(self, x):
        return self.evaluate_many([x])[0]

    def evaluate_many(self, xs):
        """Score several positions at once: every (position, fold) SVR fit runs in parallel
        on all CPU cores. Results and log order are the same as calling one by one."""
        decoded = [decode(x) for x in xs]
        folds = list(self.cv.split(self.X))
        jobs = [delayed(_fold_rmse)(params, self.X[:, mask], self.y, tr, va)
                for params, mask in decoded for tr, va in folds]
        rmse = np.array(Parallel(n_jobs=-1)(jobs)).reshape(len(xs), len(folds))
        fitnesses = []
        for (params, mask), fold_rmse in zip(decoded, rmse):
            fitness = float(fold_rmse.mean())
            best = min(fitness, self.log[-1]["best_so_far"]) if self.log else fitness
            self.log.append({"eval": len(self.log) + 1, **params, "n_features": int(mask.sum()),
                             "features": "|".join(np.array(self.names)[mask]),
                             "fitness": fitness, "best_so_far": best})
            fitnesses.append(fitness)
        return fitnesses

    @property
    def n_evals(self):
        return len(self.log)
