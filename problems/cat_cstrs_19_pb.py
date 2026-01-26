import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    # 6 binary categoricals encoded as {0,1}
    "x_cat": [[0, 1]] * 6,
    "x_int": [
        (-10, 10),  # xi1
        (-10, 10),  # xi2
    ],
    "x_con": [
        (-10.0, 10.0),  # xc1
        (-10.0, 10.0),  # xc2
    ],
}

# Coefficients (i = 1..6) stored 0-indexed
_A = np.array([2.00, 3.10, 9.80, 5.00, 3.20, 1.10], dtype=float)
_B = np.array([1.10, 0.95, 0.85, 0.60, 0.90, 0.55], dtype=float)
_C = np.array([1.40, 2.60, 9.10, 4.60, 2.05, 0.90], dtype=float)
_D = np.array([1.80, 1.55, 1.25, 1.05, 1.08, 0.85], dtype=float)


def _compute_s(X_cat: np.ndarray, xc1: np.ndarray, xc2: np.ndarray) -> np.ndarray:
    """
    Vectorized s_i = s_i(cat_i, xc1, xc2) for i=1..6.

    Returns:
        S : ndarray shape (n_samples, 6)
    """
    n = X_cat.shape[0]
    S = np.zeros((n, 6), dtype=float)

    log_term = np.log(1.0 + xc1 * xc1 + xc2 * xc2)
    exp_term = np.exp(0.02 * (xc1 - xc2))

    for i in range(6):
        is_A = (X_cat[:, i].astype(int) == 0)
        S[is_A, i] = _A[i] + _B[i] * log_term[is_A]
        S[~is_A, i] = _C[i] + _D[i] * exp_term[~is_A]

    return S


def cat_cstrs_19(X):
    """
    Cat-cstrs-19: modified G07 constrained problem.

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
    X_cat = X_cat.astype(int)
    xi1 = X_int[:, 0].astype(float)
    xi2 = X_int[:, 1].astype(float)
    xc1 = X_con[:, 0].astype(float)
    xc2 = X_con[:, 1].astype(float)

    # s1..s6
    S = _compute_s(X_cat, xc1, xc2)
    s1, s2, s3, s4, s5, s6 = [S[:, i] for i in range(6)]

    # Objective
    f = (
        s1 * s1 + s2 * s2 + s1 * s2 - 14.0 * s1 - 16.0 * s2
        + (s3 - 10.0) ** 2
        + 4.0 * (s4 - 5.0) ** 2
        + (s5 - 3.0) ** 2
        + 2.0 * (s6 - 1.0) ** 2
        + 5.0 * (xc1 ** 2)
        + 7.0 * ((xc2 - 11.0) ** 2)
        + 2.0 * ((xi1 - 10.0) ** 2)
        + ((xi2 - 7.0) ** 2)
        + 45.0
    )

    # Constraints g_i(x) <= 0
    g1 = -105.0 + 4.0 * s1 + 5.0 * s2 - 3.0 * xc1 + 9.0 * xc2
    g2 = 10.0 * s1 - 8.0 * s2 - 17.0 * xc1 + 2.0 * xc2
    g3 = -8.0 * s1 + 2.0 * s2 + 5.0 * xc1 - 2.0 * (xc2 ** 2) - 12.0

    g4 = 3.0 * ((s1 - 2.0) ** 2) + 4.0 * ((s2 - 3.0) ** 2) + 2.0 * (s3 ** 2) - 7.0 * s4 - 220.0
    g5 = 5.0 * (s1 ** 2) + 8.0 * s2 + ((s3 - 6.0) ** 2) - 2.0 * s4 - 95.0
    g6 = (s1 ** 2) + 2.0 * ((s2 - 2.0) ** 2) - 2.0 * s1 * s2 + 14.0 * s5 - 6.0 * s6 - 20.0
    g7 = 0.5 * ((s1 - 8.0) ** 2) + 2.0 * ((s2 - 4.0) ** 2) + 3.0 * (s5 ** 2) - s6 - 40.0
    g8 = -3.0 * s1 + 6.0 * s2 + 3.0 * ((xc1 - 8.0) ** 2) - 7.0 * (xc2 ** 2)

    G = np.stack([g1, g2, g3, g4, g5, g6, g7, g8], axis=1)

    for j in range(n):
        g[j] = G[j, :].tolist()

    h = np.array([np.sum(np.maximum(0.0, np.array(gj, dtype=float)) ** 2) for gj in g], dtype=float)
    return f, h, g
