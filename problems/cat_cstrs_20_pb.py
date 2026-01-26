import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    # 2 categorical vars, encoded as integers 0..9
    "x_cat": [list(range(10)), list(range(10))],
    # 2 integer vars in [-10, 25]
    "x_int": [(-10, 25), (-10, 25)],
    # 7 continuous vars with their bounds
    "x_con": [
        (0.5, 1.5),       # x1c
        (0.45, 1.35),     # x2c
        (0.5, 1.5),       # x3c
        (0.5, 1.5),       # x4c
        (0.875, 2.625),   # x5c
        (0.5, 1.5),       # x6c
        (0.5, 1.5),       # x7c
    ],
}

# S table: rows = x2^{cat} (0..9), cols = x1^{cat} (0..9)
_S = np.array(
    [
        [0.45, 0.65, 0.78, 0.87, 0.93, 0.92, 0.89, 0.81, 0.63, 0.44],
        [0.46, 0.66, 0.81, 0.90, 0.96, 0.97, 0.89, 0.80, 0.63, 0.43],
        [0.46, 0.69, 0.83, 0.96, 1.02, 1.01, 0.92, 0.85, 0.69, 0.49],
        [0.52, 0.75, 0.90, 1.02, 1.10, 1.07, 1.03, 0.93, 0.75, 0.54],
        [0.63, 0.84, 0.99, 1.12, 1.19, 1.15, 1.13, 1.01, 0.83, 0.66],
        [0.75, 0.93, 1.12, 1.23, 1.28, 1.29, 1.23, 1.12, 0.97, 0.72],
        [0.87, 1.07, 1.25, 1.34, 1.40, 1.41, 1.37, 1.26, 1.06, 0.86],
        [1.04, 1.21, 1.39, 1.50, 1.57, 1.56, 1.49, 1.39, 1.23, 1.05],
        [1.20, 1.41, 1.57, 1.67, 1.72, 1.70, 1.67, 1.56, 1.43, 1.22],
        [1.42, 1.62, 1.76, 1.87, 1.91, 1.92, 1.84, 1.77, 1.59, 1.39],
    ],
    dtype=float,
)


def cat_cstrs_20(X):
    """
    Cat-cstrs-20: modified CarSideImpact constrained problem.

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
    c1 = X_cat[:, 0].astype(int)
    c2 = X_cat[:, 1].astype(int)

    xi1 = X_int[:, 0].astype(float)
    xi2 = X_int[:, 1].astype(float)

    x1 = X_con[:, 0].astype(float)
    x2 = X_con[:, 1].astype(float)
    x3 = X_con[:, 2].astype(float)
    x4 = X_con[:, 3].astype(float)
    x5 = X_con[:, 4].astype(float)
    x6 = X_con[:, 5].astype(float)
    x7 = X_con[:, 6].astype(float)

    # s = S[row=c2, col=c1] per sample
    s = _S[c2, c1]

    # Objective (vectorized)
    f = (
        1.98
        + 4.90 * x1
        + 6.67 * x2
        + 6.98 * x3
        + 4.01 * x4
        + 1.78 * x5
        + 2.73 * xi2
        + 0.60 * s
        + 0.20 * np.abs(s - 1.20)
        + 0.12 * np.abs(x4 - 1.00)
        + 0.06 * np.abs(xi1 + 2.0)
        + 0.25 * ((x1 - 1.0) ** 2) * np.abs(x2 - 0.9)
    )

    # Constraints g_i(x) <= 0 (vectorized)
    g1 = (
        1.16
        - 0.3717 * x2 * x4
        - 0.00931 * x2 * x6
        - 0.484 * x3 * s
        + 0.01343 * xi1 * x6
        - 1.0
    )

    g2 = (
        0.261
        - 0.0159 * x1 * x2
        - 0.188 * x1 * s
        - 0.019 * x2 * xi2
        + 0.0144 * x3 * x5
        + 0.0008757 * x5 * x6
        + 0.08045 * xi1 * s
        + 0.00139 * s * x7
        + 0.000001575 * x6 * x7
        - 0.32
    )

    g3 = (
        0.214
        + 0.00817 * x5
        - 0.131 * x1 * s
        - 0.0704 * x1 * s
        + 0.03099 * x2 * xi1
        - 0.018 * x2 * xi2
        + 0.0208 * x3 * s
        + 0.121 * x3 * s
        - 0.00364 * x5 * xi1
        + 0.0007715 * x5 * x6
        - 0.0005354 * xi1 * x6
        + 0.00121 * s * x7
        - 0.32
    )

    g4 = (
        0.74
        - 0.061 * x2
        - 0.163 * x3 * s
        + 0.001232 * x3 * x6
        - 0.166 * xi2 * s
        + 0.02 * (xi2 ** 2)
        - 0.32
    )

    g5 = (
        28.98
        + 3.818 * x3
        - 4.2 * x1 * x2
        + 0.0207 * x5 * x6
        + 6.63 * xi1 * s
        - 7.7 * xi2 * s
        + 0.32 * s * x6
        - 32.0
    )

    g6 = (
        33.86
        + 2.95 * x3
        + 0.1792 * x6
        - 5.057 * x1 * x2
        - 11.02 * x2 * s
        - 0.0215 * x5 * x6
        - 9.98 * xi2 * s
        + 22.0 * (s ** 2)
        - 32.0
    )

    g7 = (
        46.36
        - 9.9 * x2
        - 12.9 * x1 * s
        + 0.1107 * x3 * x6
        - 32.0
    )

    g8 = (
        0.62
        - 0.5 * x4
        - 0.19 * x2 * x3
        - 0.0122 * x4 * x6
        + 0.009325 * xi1 * x6
        + 0.000191 * (x7 ** 2)
        - 4.0
    )

    g9 = (
        10.58
        - 0.674 * x1 * x2
        - 1.95 * x2 * s
        + 0.02054 * x3 * x6
        - 0.0198 * x4 * x6
        + 0.028 * xi1 * x6
        - 9.9
    )

    g10 = (
        16.45
        - 0.489 * x3 * xi2
        - 0.843 * x5 * xi1
        + 0.432 * s * x6
        - 0.0556 * s * x7
        - 0.000786 * (x7 ** 2)
        - 15.7
    )

    G = np.stack([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10], axis=1)

    for j in range(n):
        g[j] = G[j, :].tolist()

    h = np.array([np.sum(np.maximum(0.0, np.array(gj, dtype=float)) ** 2) for gj in g], dtype=float)
    return f, h, g
