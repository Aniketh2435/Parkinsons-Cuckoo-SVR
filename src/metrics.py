"""Regression metrics and before/after comparison tables."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    """RMSE, MAE, R^2 and MAPE (UPDRS is always > 0, so MAPE is well defined)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def compare(baseline, optimised, names=("Default SVR", "Cuckoo-SVR")):
    """Before/after table: each metric, its change, and the % improvement (positive = better)."""
    rows = []
    for m in baseline:
        b, o = baseline[m], optimised[m]
        gain = (o - b) if m == "R2" else (b - o)
        rows.append({"Metric": m, names[0]: b, names[1]: o, "Change": o - b,
                     "Improvement_%": 100 * gain / abs(b) if b else float("nan"),
                     "Better": "yes" if gain > 0 else "no"})
    return pd.DataFrame(rows).set_index("Metric")
