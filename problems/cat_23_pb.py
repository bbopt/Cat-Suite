import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    "x_cat": [
        ["A", "B", "C"],  # x_cat1
        ["D", "E"],       # x_cat2
        ["F", "G"],       # x_cat3
    ],
    "x_int": [],
    "x_con": [
        (-9.0, 12.0),   # x_con1
        (-7.0, 14.0),   # x_con2
        (-10.0, 10.0),  # x_con3
        (-8.0, 13.0),   # x_con4
        (-6.0, 15.0),   # x_con5
    ],
}

# s(x^cat) mapping: key is (x1^cat, x2^cat, x3^cat)
_S_MAP = {
    ("A", "D", "F"): 0.00,
    ("A", "D", "G"): 0.20,
    ("A", "E", "F"): -0.10,
    ("A", "E", "G"): 0.25,
    ("B", "D", "F"): 0.50,
    ("B", "D", "G"): -0.20,
    ("B", "E", "F"): -0.50,
    ("B", "E", "G"): 0.80,
    ("C", "D", "F"): 0.90,
    ("C", "D", "G"): -0.50,
    ("C", "E", "F"): 1.00,
    ("C", "E", "G"): 1.25,
}


def cat_23(X):
    """
    Cat-23: Hal04-based mixed-variable problem.

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

    c1 = X_cat[:, 0]
    c2 = X_cat[:, 1]
    c3 = X_cat[:, 2]

    X_con = X_con.astype(float)  # (n,5)

    # --- s(x^cat) mapping ---
    keys = np.stack([c1, c2, c3], axis=1)  # (n,3) strings
    s = np.array([_S_MAP[(k[0], k[1], k[2])] for k in keys], dtype=float)

    pi = np.pi

    # --- Objective ---
    f = np.zeros(X.shape[0], dtype=float)

    for i in range(1, 6):  # i = 1..5
        xi = X_con[:, i - 1]
        weight = 2.0 ** (float(i - 1) / 4.0)

        term = (
            5.0 * xi
            + 0.3 * s * (xi ** 4)
            + (1.0 - xi) ** 2
            + s * np.sin(3.0 * pi * xi + (float(i) * pi) / 5.0)
        )

        f += term * weight

    h = np.zeros(X.shape[0])
    g = [[] for _ in range(X.shape[0])]
    return f, h, g
