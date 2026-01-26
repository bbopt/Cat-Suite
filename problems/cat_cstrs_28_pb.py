import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    "x_cat": [list("ABCDEFGHIJ"), list("ABCDEFGHIJ")],   # x_cat1, x_cat2
    "x_int": [(-10, 10), (-10, 10)],                     # x_int1, x_int2
    "x_con": [(13.0, 100.0), (0.0, 100.0)],              # x_con1, x_con2
}


# s(x1_cat, x2_cat): rows = x2_cat (A..J), cols = x1_cat (A..J)
_S_TABLE = np.array([
    [11.8, 12.3, 12.9, 13.6, 14.1, 14.4, 14.6, 14.7, 14.8, 14.9],  # A
    [11.2, 12.1, 13.0, 13.9, 14.6, 15.0, 15.3, 15.5, 15.6, 15.7],  # B
    [10.8, 11.8, 13.2, 14.6, 15.5, 16.0, 16.4, 16.7, 16.9, 17.0],  # C
    [10.5, 11.6, 13.1, 15.3, 16.8, 17.6, 18.2, 18.6, 18.9, 19.1],  # D
    [10.3, 11.3, 12.8, 15.0, 17.9, 19.0, 19.8, 20.4, 20.9, 21.2],  # E
    [10.2, 11.1, 12.4, 14.5, 17.2, 20.1, 21.1, 21.9, 22.6, 23.1],  # F
    [10.1, 10.9, 12.0, 13.8, 16.3, 19.3, 22.0, 23.0, 23.8, 24.4],  # G
    [10.0, 10.8, 11.7, 13.2, 15.3, 18.2, 21.0, 23.5, 24.6, 25.4],  # H
    [10.0, 10.7, 11.4, 12.6, 14.4, 16.9, 19.6, 22.2, 24.2, 25.7],  # I
    [10.0, 10.6, 11.2, 12.1, 13.5, 15.5, 17.7, 20.1, 22.4, 25.0],  # J
], dtype=float)

_CAT_TO_IDX = {c: k for k, c in enumerate("ABCDEFGHIJ")}


def cat_cstrs_28(X):
    """
    Cat-cstrs-28: mixed-constrained G06-like problem.

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

    c1 = X_cat[:, 0]
    c2 = X_cat[:, 1]

    i1 = X_int[:, 0].astype(int)
    i2 = X_int[:, 1].astype(int)

    y1 = X_con[:, 0].astype(float)
    y2 = X_con[:, 1].astype(float)

    # s lookup (rows = c2, cols = c1)
    j1 = np.array([_CAT_TO_IDX[cc] for cc in c1], dtype=int)
    j2 = np.array([_CAT_TO_IDX[cc] for cc in c2], dtype=int)
    s = _S_TABLE[j2, j1].astype(float)

    # Objective
    f = (
        (y1 - 25.0) ** 2
        + (y2 - 40.0) ** 2
        + 6.0 * (s - 18.0) ** 2
        + 0.20 * (i1.astype(float) ** 2)
        + 0.15 * (i2.astype(float) ** 2)
        + 0.60 * np.abs(i1 - i2)
        + 0.04 * np.abs(y2 - 2.0 * s)
        + 12.0
    ).astype(float)

    # Constraints (<= 0)
    g1 = (y1 - 35.0) ** 2 + (y2 - 55.0) ** 2 - 900.0 + 20.0 * (s - 18.0) + 6.0 * i1
    g2 = (y1 - 20.0) * (y2 - 30.0) - 700.0 + 12.0 * np.abs(s - 16.0) + 2.0 * np.abs(i2)
    g3 = y2 / 100.0 + np.abs(i1) / 12.0 + 0.05 * (s - 10.0) - 1.0
    g4 = np.abs(y2 - (2.0 * s + 5.0)) - (18.0 - 0.6 * np.abs(i2))
    g5 = (i1 + i2) ** 2 - 60.0 + 4.0 * (s - 18.0)

    Gmat = np.stack([g1, g2, g3, g4, g5], axis=1).astype(float)
    for j in range(n):
        g[j] = Gmat[j, :].tolist()

    h = np.array([np.sum(np.maximum(0.0, np.array(gj, dtype=float)) ** 2) for gj in g], dtype=float)
    return f, h, g
