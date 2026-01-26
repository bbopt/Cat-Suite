import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    "x_cat": [
        ["A", "B", "C", "D", "E", "F", "G", "H"],  # x_cat1
        ["A", "B", "C", "D", "E", "F", "G", "H"],  # x_cat2
    ],
    "x_int": [],
    "x_con": [
        (2.6, 3.6),   # x1
        (0.7, 0.8),   # x2
        (7.3, 20.0),  # x3
        (0.5, 8.3),   # x4
        (2.6, 3.9),   # x5
        (5.0, 5.5),   # x6
    ],
}

# s-table: rows = x_cat2, cols = x_cat1
_S_TABLE = np.array(
    [
        [23, 19, 27, 21, 18, 24, 20, 26],  # x_cat2 = A
        [20, 25, 18, 24, 22, 19, 27, 21],  # x_cat2 = B
        [26, 22, 24, 19, 27, 21, 18, 23],  # x_cat2 = C
        [19, 27, 21, 23, 20, 26, 22, 18],  # x_cat2 = D
        [24, 18, 23, 20, 26, 22, 19, 27],  # x_cat2 = E
        [21, 23, 20, 26, 22, 18, 24, 19],  # x_cat2 = F
        [17, 24, 22, 19, 27, 21, 23, 20],  # x_cat2 = G
        [22, 19, 26, 22, 18, 23, 20, 25],  # x_cat2 = H
    ],
    dtype=float,
)

_CAT_TO_IDX = {c: i for i, c in enumerate(["A", "B", "C", "D", "E", "F", "G", "H"])}


def cat_cstrs_17(X):
    """
    Cat-cstrs-17: Speed Reducer constrained problem.

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

    cat1 = X_cat[:, 0]
    cat2 = X_cat[:, 1]

    X_con = X_con.astype(float)
    x1, x2, x3, x4, x5, x6 = [X_con[:, i] for i in range(6)]

    # --- s(x_cat1, x_cat2): rows=cat2, cols=cat1 ---
    i1 = np.array([_CAT_TO_IDX[c] for c in cat1], dtype=int)
    i2 = np.array([_CAT_TO_IDX[c] for c in cat2], dtype=int)
    s = _S_TABLE[i2, i1]

    # --- Objective ---
    poly = 3.3333 * s**2 + 14.9334 * s - 43.0934
    f = (
        0.7854 * x1 * (x2**2) * poly
        - 1.508 * x1 * (x5**2 + x6**2)
        + 7.477 * (x5**3 + x6**3)
        + 0.7854 * (x3 * (x5**2) + x4 * (x6**2))
    )

    # --- Constraints g_i(x) <= 0 ---
    g1 = 27.0 - x1 * (x2**2) * s
    g2 = 397.5 - x1 * (x2**2) * (s**2)
    g3 = 1.93 * (x3**3) - x2 * s * (x5**4)
    g4 = 1.93 * (x4**3) - x2 * s * (x6**4)

    g5 = np.sqrt((745.0 * x3) ** 2 + (16.9e6) * (x2**2) * (s**2)) - 110.0 * x2 * s * (x5**3)
    g6 = np.sqrt((745.0 * x4) ** 2 + (157.5e6) * (x2**2) * (s**2)) - 85.0 * x2 * s * (x6**3)

    g7 = x2 * s - 40.0
    g8 = 5.0 * x2 - x1
    g9 = x1 - 12.0 * x2
    g10 = 1.9 + 1.5 * x5 - x3
    g11 = 1.9 + 1.1 * x6 - x4

    G = np.stack([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11], axis=1)

    for j in range(n):
        g[j] = G[j, :].tolist()

    h = np.array([np.sum(np.maximum(0.0, np.array(gj, dtype=float)) ** 2) for gj in g], dtype=float)
    return f, h, g
