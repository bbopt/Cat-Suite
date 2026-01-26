import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    "x_cat": [list("ABCDEF"), list("ABCDEF")],   # x1_cat, x2_cat
    "x_int": [(-2, 2), (-2, 2)],                 # x1_int, x2_int
    "x_con": [(-5.0, 5.0), (-5.0, 5.0)],         # x1_con, x2_con
}


def cat_cstrs_26(X):
    """
    Cat-cstrs-26: mixed-constrained Three Hump.

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
    f = np.zeros(n, dtype=float)
    g = [[] for _ in range(n)]

    x1cat = X_cat[:, 0]
    x2cat = X_cat[:, 1]

    x1i = X_int[:, 0].astype(int)
    x2i = X_int[:, 1].astype(int)

    x1 = X_con[:, 0].astype(float)
    x2 = X_con[:, 1].astype(float)

    cat_to_idx = {c: i for i, c in enumerate("ABCDEF")}
    c1 = np.array([cat_to_idx[c] for c in x1cat], dtype=int)
    c2 = np.array([cat_to_idx[c] for c in x2cat], dtype=int)

    # ---- s1 = s(x1_cat, x1_con) ----
    s1 = np.zeros(n, dtype=float)

    idx = c1 == 0  # A
    if np.any(idx):
        z = x1[idx]
        s1[idx] = z + 0.12 * (z - 1.0) ** 2 + 0.15 * np.abs(z)

    idx = c1 == 1  # B
    if np.any(idx):
        z = x1[idx]
        s1[idx] = z + 0.12 * (z - 1.0) ** 2 + 0.10 * np.abs(z - 0.5)

    idx = c1 == 2  # C
    if np.any(idx):
        z = x1[idx]
        s1[idx] = z + 0.12 * (z - 1.0) ** 2 + 0.08 * np.abs(z + 0.5)

    idx = c1 == 3  # D
    if np.any(idx):
        z = x1[idx]
        s1[idx] = z - 0.10 * (z + 1.0) ** 2 - 0.12 * np.abs(z)

    idx = c1 == 4  # E
    if np.any(idx):
        z = x1[idx]
        s1[idx] = z - 0.10 * (z + 1.0) ** 2 - 0.09 * np.abs(z - 0.5)

    idx = c1 == 5  # F
    if np.any(idx):
        z = x1[idx]
        s1[idx] = z - 0.10 * (z + 1.0) ** 2 - 0.07 * np.abs(z + 0.5)

    # ---- s2 = s(x2_cat, x2_con) ----
    s2 = np.zeros(n, dtype=float)

    idx = c2 == 0  # A
    if np.any(idx):
        z = x2[idx]
        s2[idx] = 0.9 * np.exp(0.35 * z) + 0.05 * np.abs(z)

    idx = c2 == 1  # B
    if np.any(idx):
        z = x2[idx]
        s2[idx] = 0.6 * np.exp(0.45 * z) + 0.04 * np.abs(z)

    idx = c2 == 2  # C
    if np.any(idx):
        z = x2[idx]
        s2[idx] = 0.9 * np.exp(0.35 * z) + 0.06 * np.abs(z - 0.4)

    idx = c2 == 3  # D
    if np.any(idx):
        z = x2[idx]
        s2[idx] = 0.6 * np.exp(0.45 * z) + 0.05 * np.abs(z - 0.4)

    idx = c2 == 4  # E
    if np.any(idx):
        z = x2[idx]
        s2[idx] = 0.9 * np.exp(0.35 * z) + 0.06 * np.abs(z + 0.4)

    idx = c2 == 5  # F
    if np.any(idx):
        z = x2[idx]
        s2[idx] = 0.6 * np.exp(0.45 * z) + 0.05 * np.abs(z + 0.4)

    # ---- objective ----
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

    # ---- constraints (vectorized) ----
    g1 = ((x1 - 1.0) ** 2) / 0.10 \
         + ((x2 + 0.6 * s2 - 0.54) ** 2) / 0.06 \
         + ((x1i + 2.0 - 2.0 * np.abs(s1)) ** 2) / 9.0 \
         - 1.0

    g2 = (s2 - s1) / 0.10 \
         + np.abs(x1i) / 2.0 \
         + np.abs(x2i) / 2.0 \
         - 1.0

    g3 = ((s1 - 1.15) ** 2) / 0.05 \
         + ((s2 - 0.90) ** 2) / 0.05 \
         + np.abs(x1i + x2i) / 6.0 \
         - 1.0

    sig = 1.0 / (1.0 + np.exp(-1.2 * (x1 - x2)))
    g4 = (sig - 0.73) ** 2 \
         + (np.abs(s1) / 3.0) ** 2 \
         + (np.abs(x2i) / 10.0) ** 2 \
         - 0.12

    g5 = np.abs(x1 - x2 / 10.0 - 1.0) \
         + 0.45 * np.abs(s2 - 0.90) \
         + 0.10 * np.abs(s1 - 1.15) \
         - 0.12

    g6 = ((x2 - 0.40 * s1) ** 2) / 0.20 \
         + ((x1 + 0.30 * s2 - 1.27) ** 2) / 0.20 \
         + 0.015 * np.abs(x1i) \
         + 0.015 * np.abs(x2i) \
         - 1.0

    Gmat = np.stack([g1, g2, g3, g4, g5, g6], axis=1)
    for j in range(n):
        g[j] = Gmat[j, :].tolist()

    h = np.array([np.sum(np.maximum(0.0, np.array(gj, dtype=float)) ** 2) for gj in g], dtype=float)
    return f, h, g
