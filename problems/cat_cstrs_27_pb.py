import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    "x_cat": [[str(i) for i in range(1, 9)], [str(i) for i in range(1, 9)]],  # x1_cat, x2_cat
    "x_int": [(-10, 10), (-10, 10), (-10, 10)],                               # x1_int, x2_int, x3_int
    "x_con": [(-4.0, 4.0), (-4.0, 4.0), (-2.0, 2.0), (-2.0, 2.0)],            # x1_con, x2_con, x3_con, x4_con
}


def cat_cstrs_27(X):
    """
    Cat-cstrs-27: mixed-constrained McCormick.

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

    x1cat = X_cat[:, 0]
    x2cat = X_cat[:, 1]

    x1i = X_int[:, 0].astype(int)
    x2i = X_int[:, 1].astype(int)
    x3i = X_int[:, 2].astype(int)

    x1 = X_con[:, 0].astype(float)
    x2 = X_con[:, 1].astype(float)
    x3 = X_con[:, 2].astype(float)
    x4 = X_con[:, 3].astype(float)

    cat_to_idx = {str(i): i - 1 for i in range(1, 9)}  # "1"->0, ..., "8"->7
    c1 = np.array([cat_to_idx[c] for c in x1cat], dtype=int)
    c2 = np.array([cat_to_idx[c] for c in x2cat], dtype=int)

    # ---- helpers ----
    def sgn_arr(v: np.ndarray) -> np.ndarray:
        return np.where(v > 0.0, 1.0, np.where(v < 0.0, -1.0, 0.0))

    # ---- s1 = s(x1_cat, x1_con, x3_con) ----
    s1 = np.zeros(n, dtype=float)

    idx = c1 == 0  # "1"
    if np.any(idx):
        z1, z3 = x1[idx], x3[idx]
        s1[idx] = z1 + 0.25 * np.tanh(z1) + 0.15 * z3

    idx = c1 == 1  # "2"
    if np.any(idx):
        z1, z3 = x1[idx], x3[idx]
        s1[idx] = z1 + 0.25 * np.tanh(z1 - 0.5) + 0.12 * z3

    idx = c1 == 2  # "3"
    if np.any(idx):
        z1, z3 = x1[idx], x3[idx]
        s1[idx] = z1 + 0.25 * np.tanh(z1 + 0.5) + 0.10 * z3

    idx = c1 == 3  # "4"
    if np.any(idx):
        z1, z3 = x1[idx], x3[idx]
        s1[idx] = z1 - 0.30 * np.tanh(z1) + 0.10 * z3

    idx = c1 == 4  # "5"
    if np.any(idx):
        z1, z3 = x1[idx], x3[idx]
        s1[idx] = z1 - 0.30 * np.tanh(z1 + 0.5) + 0.08 * z3

    idx = c1 == 5  # "6"
    if np.any(idx):
        z1, z3 = x1[idx], x3[idx]
        s1[idx] = z1 - 0.30 * np.tanh(z1 - 0.5) + 0.09 * z3

    idx = c1 == 6  # "7"
    if np.any(idx):
        z1, z3 = x1[idx], x3[idx]
        s1[idx] = z1 + 0.18 * (np.abs(z1) ** 1.3) - 0.10 * z3

    idx = c1 == 7  # "8"
    if np.any(idx):
        z1, z3 = x1[idx], x3[idx]
        s1[idx] = z1 + 0.18 * (np.abs(z1 + 0.3) ** 1.3) - 0.12 * z3

    # ---- s2 = s(x2_cat, x2_con, x4_con) ----
    s2 = np.zeros(n, dtype=float)

    idx = c2 == 0  # "1"
    if np.any(idx):
        z2, z4 = x2[idx], x4[idx]
        s2[idx] = z2 + 0.35 * np.arctan(z2) + 0.20 * z4

    idx = c2 == 1  # "2"
    if np.any(idx):
        z2, z4 = x2[idx], x4[idx]
        s2[idx] = z2 + 0.35 * np.arctan(z2 - 0.6) + 0.18 * z4

    idx = c2 == 2  # "3"
    if np.any(idx):
        z2, z4 = x2[idx], x4[idx]
        s2[idx] = z2 + 0.35 * np.arctan(z2 + 0.6) + 0.16 * z4

    idx = c2 == 3  # "4"
    if np.any(idx):
        z2, z4 = x2[idx], x4[idx]
        s2[idx] = z2 + 0.22 * sgn_arr(z2) * np.sqrt(np.abs(z2)) - 0.12 * z4

    idx = c2 == 4  # "5"
    if np.any(idx):
        z2, z4 = x2[idx], x4[idx]
        s2[idx] = z2 + 0.22 * sgn_arr(z2) * np.sqrt(np.abs(z2 + 0.4)) - 0.10 * z4

    idx = c2 == 5  # "6"
    if np.any(idx):
        z2, z4 = x2[idx], x4[idx]
        s2[idx] = z2 + 0.22 * sgn_arr(z2) * np.sqrt(np.abs(z2 - 0.4)) - 0.11 * z4

    idx = c2 == 6  # "7"
    if np.any(idx):
        z2, z4 = x2[idx], x4[idx]
        s2[idx] = z2 - 0.28 * np.log(1.0 + z2 * z2) + 0.15 * z4

    idx = c2 == 7  # "8"
    if np.any(idx):
        z2, z4 = x2[idx], x4[idx]
        s2[idx] = z2 - 0.28 * np.log(1.0 + (z2 - 0.4) ** 2) + 0.13 * z4

    # ---- objective ----
    f = (
        np.sin(s1 + s2)
        + (s1 - s2) ** 2
        - 1.5 * s1
        + 2.5 * s2
        + 1.0
        + 0.06 * np.abs(s1 + x1i / 10.0)
        + 0.04 * np.abs(s2 - x2i / 10.0)
        + 0.03 * np.abs(s1 - s2 + x3i / 10.0)
    ).astype(float)

    # ---- constraints ----
    g1 = (
        ((s1 - 0.80) ** 2) / 0.55
        + ((s2 - 0.55) ** 2) / 0.60
        + ((x3 - 0.40) ** 2) / 0.90
        + ((x1i - 4.0) ** 2) / 36.0
        - 1.0
    )

    g2 = (
        np.abs(np.sin(s1 + s2 - 0.35))
        + np.abs(x2i + 2.0) / 20.0
        + 0.30 * np.maximum(0.0, np.abs(s2 - s1) - 0.35)
        - 0.80
    )

    logistic = 1.0 / (1.0 + np.exp(-1.1 * (x2 - x1 - 0.4)))
    g3 = (
        (logistic - 0.60) ** 2
        + ((x4 + 0.6) / 1.8) ** 2
        + np.abs(x3i - 1.0) / 25.0
        + 0.10 / (1.0 + np.abs(s1))
        + 0.08 / (1.0 + np.abs(s2))
        - 0.22
    )

    Gmat = np.stack([g1, g2, g3], axis=1)
    for j in range(n):
        g[j] = Gmat[j, :].tolist()

    h = np.array([np.sum(np.maximum(0.0, np.array(gj, dtype=float)) ** 2) for gj in g], dtype=float)
    return f, h, g
