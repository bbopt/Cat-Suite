import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    "x_cat": [list("ABCDEFGHIJ")],      # x_cat
    "x_int": [],                        # none
    "x_con": [(0.05, 1.00), (0.05, 1.00)],  # x1, x2
}


def cat_cstrs_25(X):
    """
    Cat-cstrs-25: modified Three-Bar Truss (constrained).

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

    cat = X_cat[:, 0]
    x1 = X_con[:, 0].astype(float)
    x2 = X_con[:, 1].astype(float)

    # constants
    H = 100.0
    P = 2.0
    sigma = 2.0
    eps = 1e-12

    cat_to_idx = {c: i for i, c in enumerate("ABCDEFGHIJ")}
    cat_idx = np.array([cat_to_idx[c] for c in cat], dtype=int)

    # --- s(x_cat, x1, x2) ---
    s = np.zeros(n, dtype=float)

    # A
    idx = cat_idx == 0
    if np.any(idx):
        inner = np.abs(x1[idx] - 0.70) + 0.30 * np.abs(x2[idx] - 0.35) - 0.25
        s[idx] = 50.0 * inner

    # B
    idx = cat_idx == 1
    if np.any(idx):
        inner = (x1[idx] - 0.80) ** 2 + 0.60 * np.abs(x2[idx] - 0.40) - 0.18
        s[idx] = 50.0 * inner

    # C
    idx = cat_idx == 2
    if np.any(idx):
        inner = np.abs(x1[idx] + x2[idx] - 1.15) + 0.25 * (x2[idx] - 0.45) ** 2 - 0.22
        s[idx] = 50.0 * inner

    # D
    idx = cat_idx == 3
    if np.any(idx):
        inner = np.maximum(0.0, x2[idx] - 0.55) ** 2 + 0.40 * np.abs(x1[idx] - 0.75) - 0.20
        s[idx] = 50.0 * inner

    # E
    idx = cat_idx == 4
    if np.any(idx):
        inner = np.abs(x1[idx] * x2[idx] - 0.30) + 0.15 * np.abs(x1[idx] - x2[idx]) - 0.15
        s[idx] = 50.0 * inner

    # F  (careful with |x1-0.65|)
    idx = cat_idx == 5
    if np.any(idx):
        denom = np.maximum(np.abs(x1[idx] - 0.65), eps)
        inner = ((x2[idx] - 0.25) ** 2) / denom - 0.10
        s[idx] = 50.0 * inner

    # G  (careful with |x2-0.35|)
    idx = cat_idx == 6
    if np.any(idx):
        denom = np.maximum(np.abs(x2[idx] - 0.35), eps)
        inner = ((x1[idx] - 0.85) ** 2) / denom - 0.10
        s[idx] = 50.0 * inner

    # H
    idx = cat_idx == 7
    if np.any(idx):
        inner = np.abs(x1[idx] - 0.90) + np.abs(x2[idx] - 0.20) - 0.30
        s[idx] = 50.0 * inner

    # I
    idx = cat_idx == 8
    if np.any(idx):
        inner = np.maximum(0.0, 1.05 - x1[idx] - x2[idx]) ** 2 - 0.08
        s[idx] = 50.0 * inner

    # J
    idx = cat_idx == 9
    if np.any(idx):
        inner = 0.50 * np.abs(x1[idx] - 0.78) + 0.50 * np.abs(x2[idx] - 0.42) - 0.19
        s[idx] = 50.0 * inner

    # --- objective ---
    f = (2.0 * np.sqrt(2.0) * x1 + x2) * H * (1.0 + 1.0e-3 * s)

    # --- constraints ---
    denom = np.sqrt(2.0 * x1 * x1 + 2.0 * x1 * x2)
    denom = np.maximum(denom, eps)

    g1 = ((np.sqrt(2.0) * x1 + x2) / denom) * P - sigma + 0.002 * s - 0.04
    g2 = (x2 / denom) * P - sigma + 0.0015 * np.abs(s)
    g3 = (1.0 / np.maximum(x1 + np.sqrt(2.0) * x2, eps)) * P - sigma + 0.001 * (s * s)

    Gmat = np.stack([g1, g2, g3], axis=1)
    for j in range(n):
        g[j] = Gmat[j, :].tolist()

    h = np.array([np.sum(np.maximum(0.0, np.array(gj, dtype=float)) ** 2) for gj in g], dtype=float)
    return f, h, g
