import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    "x_cat": [["A", "B", "C", "D", "E", "F", "G"]] * 2,   # x_cat1, x_cat2
    "x_int": [(1, 20)] * 1,                               # x_int
    "x_con": [(-1.0, 1.0)] * 7,                           # x1..x7
}


def cat_cstrs_30(X):
    """
    Cat-cstrs-30: constrained Ishigami-like mixed problem.

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

    # cats (strings)
    c1 = X_cat[:, 0]
    c2 = X_cat[:, 1]

    # int
    xi = X_int[:, 0].astype(int)

    # cons
    x1 = X_con[:, 0].astype(float)
    x2 = X_con[:, 1].astype(float)
    x3 = X_con[:, 2].astype(float)
    x4 = X_con[:, 3].astype(float)
    x5 = X_con[:, 4].astype(float)
    x6 = X_con[:, 5].astype(float)
    x7 = X_con[:, 6].astype(float)

    cats = ["A", "B", "C", "D", "E", "F", "G"]
    cat_to_idx = {c: k for k, c in enumerate(cats)}

    # s(x1^cat, x2^cat): rows = x2^cat, cols = x1^cat
    Smap = np.array(
        [
            [0.40, 0.66, 0.91, 1.06, 0.89, 0.61, 0.37],  # x2=A
            [0.56, 0.81, 1.07, 1.26, 1.09, 0.78, 0.52],  # x2=B
            [0.71, 0.97, 1.22, 1.43, 1.24, 0.94, 0.69],  # x2=C
            [0.91, 1.17, 1.43, 1.67, 1.47, 1.14, 0.85],  # x2=D
            [1.12, 1.37, 1.63, 1.92, 1.70, 1.31, 1.03],  # x2=E
            [1.36, 1.61, 1.94, 2.23, 1.96, 1.57, 1.26],  # x2=F
            [1.61, 1.93, 2.24, 2.53, 2.22, 1.82, 1.51],  # x2=G
        ],
        dtype=float,
    )

    j1 = np.array([cat_to_idx[c] for c in c1], dtype=int)  # cols
    j2 = np.array([cat_to_idx[c] for c in c2], dtype=int)  # rows
    s = Smap[j2, j1].astype(float)

    pi = np.pi

    # Objective
    term1 = (s ** 2) * (
        np.sin(pi * x1)
        + 7.0 * (np.sin(pi * x2) ** 2.0)
        + 0.1 * ((pi * x3) ** 4.0) * np.sin(pi * x1)
    )

    term2 = s * (
        (x4 ** 2 + x5 ** 2 - 0.5)
        + (xi.astype(float) / 20.0) * np.sin(4.0 * (x4 + x5))
    )

    term3 = 0.3 * (
        ((x6 ** 2 + x7 ** 2 - 1.0) ** 2.0)
        + (xi.astype(float) / 20.0) * np.cos(4.0 * (x6 - x7))
    )

    f = (term1 + term2 + term3).astype(float)

    # Constraints (<= 0)
    g1 = ((x4 ** 2 + x5 ** 2) - (0.35 + 0.25 * s)) ** 2.0 - 0.0064
    g2 = ((x6 ** 2 + x7 ** 2) - (0.55 + 0.15 * s)) ** 2.0 - 0.0100

    expr3 = (
        np.sin(pi * x1)
        + 0.5 * np.sin(pi * x2)
        + 0.25 * x3
        + (xi.astype(float) - 10.0) / 20.0
        - 0.15 * s
    )
    g3 = (expr3 ** 2.0) - 0.0064

    Gmat = np.stack([g1, g2, g3], axis=1).astype(float)
    for j in range(n):
        g[j] = Gmat[j, :].tolist()

    h = np.array([np.sum(np.maximum(0.0, np.array(gj, dtype=float)) ** 2) for gj in g], dtype=float)
    return f, h, g
