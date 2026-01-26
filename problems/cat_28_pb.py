import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    "x_cat": [
        ["A", "B", "C", "D", "E", "F"],  # x_cat1
        ["A", "B", "C", "D", "E", "F"],  # x_cat2
    ],
    "x_int": [],
    "x_con": [
        (-5.0, 5.0),  # x_con1
        (-5.0, 5.0),  # x_con2
    ],
}


def _s1(cat1: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """
    Vectorized s1 = s(x1^{cat}, x1^{con}).
    cat1 shape: (n_samples,)
    x1 shape: (n_samples,)
    """
    x1 = np.asarray(x1, dtype=float)
    out = np.empty_like(x1, dtype=float)

    m = (cat1 == "A")
    out[m] = x1[m] + 0.12 * (x1[m] - 1.0) ** 2 + 0.15 * np.abs(x1[m])

    m = (cat1 == "B")
    out[m] = x1[m] + 0.12 * (x1[m] - 1.0) ** 2 + 0.10 * np.abs(x1[m] - 0.5)

    m = (cat1 == "C")
    out[m] = x1[m] + 0.12 * (x1[m] - 1.0) ** 2 + 0.08 * np.abs(x1[m] + 0.5)

    m = (cat1 == "D")
    out[m] = x1[m] - 0.10 * (x1[m] + 1.0) ** 2 - 0.12 * np.abs(x1[m])

    m = (cat1 == "E")
    out[m] = x1[m] - 0.10 * (x1[m] + 1.0) ** 2 - 0.09 * np.abs(x1[m] - 0.5)

    m = (cat1 == "F")
    out[m] = x1[m] - 0.10 * (x1[m] + 1.0) ** 2 - 0.07 * np.abs(x1[m] + 0.5)

    return out


def _s2(cat2: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Vectorized s2 = s(x2^{cat}, x2^{con}).
    cat2 shape: (n_samples,)
    x2 shape: (n_samples,)
    """
    x2 = np.asarray(x2, dtype=float)
    out = np.empty_like(x2, dtype=float)

    m = (cat2 == "A")
    out[m] = 0.9 * np.exp(0.35 * x2[m]) + 0.05 * np.abs(x2[m])

    m = (cat2 == "B")
    out[m] = 0.6 * np.exp(0.45 * x2[m]) + 0.04 * np.abs(x2[m])

    m = (cat2 == "C")
    out[m] = 0.9 * np.exp(0.35 * x2[m]) + 0.06 * np.abs(x2[m] - 0.4)

    m = (cat2 == "D")
    out[m] = 0.6 * np.exp(0.45 * x2[m]) + 0.05 * np.abs(x2[m] - 0.4)

    m = (cat2 == "E")
    out[m] = 0.9 * np.exp(0.35 * x2[m]) + 0.06 * np.abs(x2[m] + 0.4)

    m = (cat2 == "F")
    out[m] = 0.6 * np.exp(0.45 * x2[m]) + 0.05 * np.abs(x2[m] + 0.4)

    return out


def cat_28(X):
    """
    Cat-28: Three-hump based mixed-variable problem.

    Parameters:
        X : ndarray of shape (n_samples, n_variables)

    Returns:
        f : ndarray of shape (n_samples,)
        h : ndarray of shape (n_samples,) — zero for unconstrained problems
        g : list of empty lists — one per sample
    """
    X = np.atleast_2d(X)
    X_cat, X_int, X_con = split_into_components(X, BOUNDS)
    check_bounds(X_cat, X_int, X_con, BOUNDS)

    cat1 = X_cat[:, 0]
    cat2 = X_cat[:, 1]

    X_con = X_con.astype(float)
    x1 = X_con[:, 0]
    x2 = X_con[:, 1]

    s1 = _s1(cat1, x1)
    s2 = _s2(cat2, x2)

    s1_2 = s1 * s1
    s1_4 = s1_2 * s1_2
    s1_6 = s1_4 * s1_2

    f = (
        2.0 * s1_2
        - 1.05 * s1_4
        + (s1_6 / 6.0)
        + s1 * s2
        + s2 * s2
        + 0.08 * np.abs(s1)
        + 0.05 * np.abs(s2 - s1)
    )

    h = np.zeros(X.shape[0])
    g = [[] for _ in range(X.shape[0])]
    return f, h, g
