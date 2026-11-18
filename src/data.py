"""Loading, cleaning and splitting the Oxford Parkinson's Disease Telemonitoring dataset.

5,875 voice recordings from 42 early-stage Parkinson's patients. Each row has the
patient's age and sex, 16 biomedical voice measures, and the clinician-rated
motor UPDRS score (the regression target).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "motor_UPDRS"
# Identifiers, the time index and the other score are not voice features.
NON_FEATURES = ["subject#", "test_time", "motor_UPDRS", "total_UPDRS"]
# Jitter, shimmer and NHR are strongly right-skewed ratios, so they are log-transformed.
SKEWED_PREFIXES = ("Jitter", "Shimmer", "NHR")


@dataclass
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list
    frame: pd.DataFrame                    # cleaned frame (before log transform), for EDA
    report: dict = field(default_factory=dict)

    @property
    def n_features(self):
        return len(self.feature_names)


def load_raw(path):
    return pd.read_csv(path)


def clean(df):
    """Drop exact duplicate rows and rows with missing values; report what was removed."""
    report = {"rows_raw": len(df), "missing_values": int(df.isna().sum().sum()),
              "duplicate_rows": int(df.duplicated().sum())}
    df = df.drop_duplicates().dropna().reset_index(drop=True)
    report["rows_clean"] = len(df)
    report["patients"] = int(df["subject#"].nunique())
    return df, report


def log_transform(X):
    """log1p on the skewed voice measures (stateless, so it cannot leak test information)."""
    X = X.copy()
    cols = [c for c in X.columns if c.startswith(SKEWED_PREFIXES)]
    X[cols] = np.log1p(X[cols])
    return X, cols


def prepare_data(path, test_size=0.2, seed=42):
    """Load -> clean -> log-transform skewed features -> 80/20 train/test split.

    Feature scaling happens inside each model's pipeline, fitted on training folds only.
    """
    df, report = clean(load_raw(path))
    X, logged = log_transform(df.drop(columns=NON_FEATURES))
    y = df[TARGET].values
    X_tr, X_te, y_tr, y_te = train_test_split(X.values, y, test_size=test_size, random_state=seed)
    report.update(features=X.shape[1], log_transformed=logged, train=len(y_tr), test=len(y_te),
                  target_mean=float(y.mean()), target_std=float(y.std()),
                  target_range=[float(y.min()), float(y.max())])
    return Dataset(X_tr, X_te, y_tr, y_te, list(X.columns), df, report)
