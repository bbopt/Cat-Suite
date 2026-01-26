import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    # x_cat is the single categorical variable "k" with options 1..50
    "x_cat": [list(range(1, 51))],
    "x_int": [
        (-2, 2),  # xi1
        (-2, 2),  # xi2
    ],
    "x_con": [
        (0.25, 1.3),  # x1c
        (0.05, 2.0),  # x2c
    ],
}


def _s_from_cat(k: np.ndarray) -> np.ndarray:
    """
    Vectorized s = s(k).
    k is numeric with values in {1,...,50}.
    """
    k = k.astype(float)
    even = (np.mod(k, 2.0) == 0.0)
    s = np.empty_like(k, dtype=float)

    s[even] = 10.0 + 0.3 * k[even] + 0.7 * np.sin(np.pi * k[even] / 6.0)
    s[~even] = 25.0 * np.exp(-0.05 * k[~even]) + 1.5
    return s


def cat_cstrs_18(X):
    """
    Cat-cstrs-18: modified Spring constrained problem.

    Parameters:
        X : ndarray of shape (n_samples, n_variables)

    Returns:
        f : ndarray of shape (n_samples,)
        h : ndarray of shape (n_samples,)
        g : list of lists — individual constraint values per sample
    """
    X = np.atleast_2d(X)
    X_cat, X_int, X_con = split_into_components(X, BOUNDS)
    check_bounds(X_cat, X_int, X_con, BOUNDS)

    n = X.shape[0]
    g = [[] for _ in range(n)]

    # Extract
    k = X_cat[:, 0].astype(float)          # {1..50} (numeric categorical)
    xi1 = X_int[:, 0].astype(float)
    xi2 = X_int[:, 1].astype(float)
    x1c = X_con[:, 0].astype(float)
    x2c = X_con[:, 1].astype(float)

    s = _s_from_cat(k)

    # Objective
    f = (
        (s + 2.0) * x1c * (x2c**2)
        + 0.05 * xi1 * (x1c**2)
        + 0.03 * (xi2**2) * x2c
        + 0.01 * xi1 * xi2 * s
    )

    # Constraints g_i(x) <= 0
    g1 = 71785.0 * (x2c**4) - (x1c**3) * s + 500.0 * xi1 * xi2

    g2 = (
        5108.0 * (x2c**2) * (4.0 * (x1c**2) - x1c * x2c)
        + 12566.0 * (x1c * (x2c**3) - (x1c**4))
        - 64187128.0 * (x2c**5) * (x1c - x2c)
        + 2000.0 * ((xi1**2) - xi2)
    )

    g3 = (x1c**2) * s - 140.45 * x2c + 10.0 * xi1
    g4 = x1c + x2c - 1.5
    g5 = xi1 + 2.0 * xi2 - 15.0

    G = np.stack([g1, g2, g3, g4, g5], axis=1)

    for j in range(n):
        g[j] = G[j, :].tolist()

    h = np.array([np.sum(np.maximum(0.0, np.array(gj, dtype=float)) ** 2) for gj in g], dtype=float)
    return f, h, g
