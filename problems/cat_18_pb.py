import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    "x_cat": [
        ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],  # x_cat1
        ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],  # x_cat2
    ],
    "x_int": [
        (0, 17),  # x_int
    ],
    "x_con": [
        (0.0, 1.0),  # x_con1
        (0.0, 1.0),  # x_con2
        (0.0, 1.0),  # x_con3
        (0.0, 1.0),  # x_con4
        (0.0, 1.0),  # x_con5
        (0.0, 1.0),  # x_con6
        (0.0, 1.0),  # x_con7
    ],
}

# s(x1^cat, x2^cat): rows = x2, cols = x1
_S_TABLE = np.array(
    [
        [0.41, 0.52, 0.59, 0.70, 0.76, 0.79, 0.73, 0.68, 0.61, 0.47],
        [0.50, 0.61, 0.71, 0.81, 0.88, 0.92, 0.87, 0.80, 0.69, 0.57],
        [0.57, 0.69, 0.81, 0.93, 1.01, 1.05, 0.99, 0.91, 0.79, 0.65],
        [0.64, 0.77, 0.89, 1.04, 1.13, 1.18, 1.11, 1.03, 0.91, 0.73],
        [0.71, 0.86, 1.01, 1.16, 1.26, 1.31, 1.24, 1.14, 0.99, 0.81],
        [0.78, 0.95, 1.11, 1.27, 1.38, 1.44, 1.37, 1.26, 1.09, 0.89],
        [0.85, 1.03, 1.21, 1.39, 1.51, 1.57, 1.49, 1.37, 1.21, 0.97],
        [0.92, 1.12, 1.31, 1.50, 1.63, 1.70, 1.62, 1.49, 1.29, 1.05],
        [0.99, 1.20, 1.41, 1.62, 1.76, 1.83, 1.74, 1.60, 1.39, 1.13],
        [1.06, 1.29, 1.51, 1.73, 1.88, 1.96, 1.87, 1.72, 1.49, 1.21],
    ],
    dtype=float,
)

_CAT_TO_IDX = {c: i for i, c in enumerate(BOUNDS["x_cat"][0])}

# Hartmann constants (exactly as PyMoo/NOMAD)
_ALPHA = np.array([1.0, 1.2, 3.0, 3.2], dtype=float)
_A = np.array(
    [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ],
    dtype=float,
)
_P = np.array(
    [
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
    ],
    dtype=float,
)


def cat_18(X):
    """
    Hartmann-based mixed-variable problem (with categorical table s).

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

    # integer in {0..17}
    x_int = X_int[:, 0].astype(float)

    X_con = X_con.astype(float)
    x1 = X_con[:, 0]
    x2 = X_con[:, 1]
    x3 = X_con[:, 2]
    x4 = X_con[:, 3]
    x5 = X_con[:, 4]
    x6 = X_con[:, 5]
    x7 = X_con[:, 6]

    # --- s(x1^cat, x2^cat): rows=x2, cols=x1 ---
    i1 = np.vectorize(_CAT_TO_IDX.get)(cat1)
    i2 = np.vectorize(_CAT_TO_IDX.get)(cat2)
    s = _S_TABLE[i2, i1]

    # --- Hartmann(6D) part using x1..x6 (vectorized over samples) ---
    X6 = np.stack([x1, x2, x3, x4, x5, x6], axis=1)          # (n,6)
    d = X6[:, None, :] - _P[None, :, :]                      # (n,4,6)
    inner = np.sum(_A[None, :, :] * (d**2), axis=2)          # (n,4)
    hartmann = np.sum(_ALPHA[None, :] * np.exp(-inner), axis=1)  # (n,)

    main_bracket = -hartmann
    scale = s * (1.0 + 0.02 * x_int)

    # --- Sin perturbation term uses x7 and x_int ---
    pi = np.pi
    phase = 4.0 * pi * x7 + (2.0 * pi * x_int) / 17.0
    term2 = 0.1 * s * np.sin(phase)

    f = scale * main_bracket + term2

    h = np.zeros(X.shape[0])
    g = [[] for _ in range(X.shape[0])]
    return f, h, g
