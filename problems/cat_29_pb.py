import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    "x_cat": [
        ["A", "B", "C", "D", "E", "F", "G", "H"],  # x_cat1
        ["A", "B", "C", "D", "E", "F", "G", "H"],  # x_cat2
    ],
    "x_int": [
        (-10, 10),  # x_int1
        (-10, 10),  # x_int2
        (-10, 10),  # x_int3
    ],
    "x_con": [
        (-4.0, 4.0),  # x_con1
        (-4.0, 4.0),  # x_con2
        (-2.0, 2.0),  # x_con3
        (-2.0, 2.0),  # x_con4
    ],
}


def _sgn(v: np.ndarray) -> np.ndarray:
    """Vectorized sign used in the PyMoo code: 1 if >0, -1 if <0, 0 if ==0."""
    v = np.asarray(v, dtype=float)
    return np.where(v > 0.0, 1.0, np.where(v < 0.0, -1.0, 0.0))


def _s1(cat1: np.ndarray, x1c: np.ndarray, x3c: np.ndarray) -> np.ndarray:
    """
    Vectorized s1 = s(x1^cat, x1^con, x3^con).
    """
    x1c = np.asarray(x1c, dtype=float)
    x3c = np.asarray(x3c, dtype=float)
    out = np.empty_like(x1c, dtype=float)

    m = (cat1 == "A")
    out[m] = x1c[m] + 0.25 * np.tanh(x1c[m]) + 0.15 * x3c[m]

    m = (cat1 == "B")
    out[m] = x1c[m] + 0.25 * np.tanh(x1c[m] - 0.5) + 0.12 * x3c[m]

    m = (cat1 == "C")
    out[m] = x1c[m] + 0.25 * np.tanh(x1c[m] + 0.5) + 0.10 * x3c[m]

    m = (cat1 == "D")
    out[m] = x1c[m] - 0.30 * np.tanh(x1c[m]) + 0.10 * x3c[m]

    m = (cat1 == "E")
    out[m] = x1c[m] - 0.30 * np.tanh(x1c[m] + 0.5) + 0.08 * x3c[m]

    m = (cat1 == "F")
    out[m] = x1c[m] - 0.30 * np.tanh(x1c[m] - 0.5) + 0.09 * x3c[m]

    m = (cat1 == "G")
    out[m] = x1c[m] + 0.18 * (np.abs(x1c[m]) ** 1.3) - 0.10 * x3c[m]

    m = (cat1 == "H")
    out[m] = x1c[m] + 0.18 * (np.abs(x1c[m] + 0.3) ** 1.3) - 0.12 * x3c[m]

    return out


def _s2(cat2: np.ndarray, x2c: np.ndarray, x4c: np.ndarray) -> np.ndarray:
    """
    Vectorized s2 = s(x2^cat, x2^con, x4^con).
    """
    x2c = np.asarray(x2c, dtype=float)
    x4c = np.asarray(x4c, dtype=float)
    out = np.empty_like(x2c, dtype=float)

    m = (cat2 == "A")
    out[m] = x2c[m] + 0.35 * np.arctan(x2c[m]) + 0.20 * x4c[m]

    m = (cat2 == "B")
    out[m] = x2c[m] + 0.35 * np.arctan(x2c[m] - 0.6) + 0.18 * x4c[m]

    m = (cat2 == "C")
    out[m] = x2c[m] + 0.35 * np.arctan(x2c[m] + 0.6) + 0.16 * x4c[m]

    m = (cat2 == "D")
    out[m] = x2c[m] + 0.22 * _sgn(x2c[m]) * np.sqrt(np.abs(x2c[m])) - 0.12 * x4c[m]

    m = (cat2 == "E")
    out[m] = x2c[m] + 0.22 * _sgn(x2c[m]) * np.sqrt(np.abs(x2c[m] + 0.4)) - 0.10 * x4c[m]

    m = (cat2 == "F")
    out[m] = x2c[m] + 0.22 * _sgn(x2c[m]) * np.sqrt(np.abs(x2c[m] - 0.4)) - 0.11 * x4c[m]

    m = (cat2 == "G")
    out[m] = x2c[m] - 0.28 * np.log(1.0 + x2c[m] * x2c[m]) + 0.15 * x4c[m]

    m = (cat2 == "H")
    out[m] = x2c[m] - 0.28 * np.log(1.0 + (x2c[m] - 0.4) ** 2) + 0.13 * x4c[m]

    return out


def cat_29(X):
    """
    Cat-29: McCormick-based mixed-variable problem.

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

    X_int = X_int.astype(float)
    x1i = X_int[:, 0]
    x2i = X_int[:, 1]
    x3i = X_int[:, 2]

    X_con = X_con.astype(float)
    x1c = X_con[:, 0]
    x2c = X_con[:, 1]
    x3c = X_con[:, 2]
    x4c = X_con[:, 3]

    s1 = _s1(cat1, x1c, x3c)
    s2 = _s2(cat2, x2c, x4c)

    f = (
        np.sin(s1 + s2)
        + (s1 - s2) ** 2
        - 1.5 * s1
        + 2.5 * s2
        + 1.0
        + 0.06 * np.abs(s1 + x1i / 10.0)
        + 0.04 * np.abs(s2 - x2i / 10.0)
        + 0.03 * np.abs(s1 - s2 + x3i / 10.0)
    )

    h = np.zeros(X.shape[0])
    g = [[] for _ in range(X.shape[0])]
    return f, h, g
