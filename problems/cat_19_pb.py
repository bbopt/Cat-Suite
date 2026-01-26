import numpy as np
from _utils import split_into_components, check_bounds

# -----------------------
# Problem metadata
# -----------------------
# Note: the PyMoo code uses a global Ncon=2. This template matches that (n_con = 2).
NCON = 2

BOUNDS = {
    "x_cat": [
        ["A", "B", "C", "D", "E", "F", "G", "H"],  # x_cat1
        ["A", "B", "C", "D", "E", "F", "G", "H"],  # x_cat2
    ],
    "x_int": [],
    "x_con": [(-10.0, 10.0)] * NCON,
}

_CAT_TO_IDX = {c: i for i, c in enumerate(BOUNDS["x_cat"][0])}


def _s1(cat_idx: int, y1: np.ndarray) -> np.ndarray:
    """
    Vectorized version of LevyProblem._s1.
    y1 shape: (n_samples,)
    """
    y1 = np.asarray(y1, dtype=float)

    out = np.empty_like(y1, dtype=float)

    m = (cat_idx == 0)  # A
    out[m] = y1[m] * y1[m] + 0.10

    m = (cat_idx == 1)  # B
    out[m] = (y1[m] - 0.25) ** 2 + 0.33

    m = (cat_idx == 2)  # C
    out[m] = (y1[m] - 0.50) ** 2 + 0.20

    m = (cat_idx == 3)  # D
    out[m] = 0.8 * (y1[m] - 0.75) ** 2 + 0.25

    m = (cat_idx == 4)  # E
    out[m] = 1.2 * (y1[m] - 1.00) ** 2 + 0.15

    m = (cat_idx == 5)  # F
    out[m] = (y1[m] - 1.25) ** 2 + 0.40

    m = (cat_idx == 6)  # G
    out[m] = 0.9 * (y1[m] - 1.50) ** 2 + 0.30

    m = (cat_idx == 7)  # H
    out[m] = 1.1 * (y1[m] - 0.00) ** 2 + 0.05

    return out


def _s2(cat_idx: int, yN: np.ndarray) -> np.ndarray:
    """
    Vectorized version of LevyProblem._s2.
    yN shape: (n_samples,)
    """
    yN = np.asarray(yN, dtype=float)

    out = np.empty_like(yN, dtype=float)

    m = (cat_idx == 0)  # A
    out[m] = (yN[m] - 1.0) ** 2

    m = (cat_idx == 1)  # B
    out[m] = 0.8 * (yN[m] - 1.0) ** 2 + 0.05

    m = (cat_idx == 2)  # C
    out[m] = 1.1 * (yN[m] - 1.05) ** 2 + 0.08

    m = (cat_idx == 3)  # D
    out[m] = 0.9 * (yN[m] - 0.95) ** 2 + 0.10

    m = (cat_idx == 4)  # E
    out[m] = -np.exp(-(yN[m] - 0.5) ** 2) + 1.10

    m = (cat_idx == 5)  # F
    out[m] = -0.9 * np.exp(-(yN[m] - 0.7) ** 2) + 1.05

    m = (cat_idx == 6)  # G
    out[m] = -1.1 * np.exp(-(yN[m] - 0.3) ** 2) + 1.15

    m = (cat_idx == 7)  # H
    out[m] = -np.exp(-(yN[m] - 0.9) ** 2) + 1.00

    return out


def cat_19(X):
    """
    Cat-19: modified Levy (with NCON=2 as in the provided PyMoo code).

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
    X_con = X_con.astype(float)  # (n_samples, NCON)

    # --- Build y_i = 1 + (x_i - 1)/4 ---
    y = 1.0 + (X_con - 1.0) / 4.0

    pi = np.pi

    # --- s1 and s2 ---
    c1 = np.vectorize(_CAT_TO_IDX.get)(cat1)
    c2 = np.vectorize(_CAT_TO_IDX.get)(cat2)

    y1 = y[:, 0]
    yN = y[:, -1]

    s1 = _s1(c1, y1)
    s2 = _s2(c2, yN)

    # --- Main sums (match NOMAD loops) ---
    sum1 = np.zeros(X.shape[0], dtype=float)
    sum2 = np.zeros(X.shape[0], dtype=float)

    for i in range(NCON - 1):
        yi = y[:, i]
        yip1 = y[:, i + 1]

        t = yi - 1.0
        sin_term = np.sin(pi * yip1)

        sum1 += (t * t) * (1.0 + 0.5 * sin_term * sin_term)
        sum2 += (yi - 1.0) * ((yip1 + 2.0) ** 3)

    f = s1 + sum1 + s2 + 0.1 * sum2

    h = np.zeros(X.shape[0])
    g = [[] for _ in range(X.shape[0])]
    return f, h, g
