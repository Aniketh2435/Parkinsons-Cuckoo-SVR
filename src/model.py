"""Support Vector Regression model and the non-optimised baselines."""
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# scikit-learn's own SVR defaults: the "no optimisation" baseline.
DEFAULT_PARAMS = {"C": 1.0, "gamma": "scale", "epsilon": 0.1}


def build_svr(C=1.0, gamma="scale", epsilon=0.1):
    """RBF-kernel SVR with feature scaling and target scaling.

    Scaling the target makes C and epsilon independent of the UPDRS units, so the same
    search ranges work for any target, and it gives the default SVR a fair chance.
    """
    svr = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon))
    return TransformedTargetRegressor(regressor=svr, transformer=StandardScaler())


def baselines():
    """Reference models trained on all features with no tuning."""
    return {
        "Mean predictor": DummyRegressor(strategy="mean"),
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Default SVR": build_svr(**DEFAULT_PARAMS),
    }
