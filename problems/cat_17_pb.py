import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    "x_cat": [
        ["A", "B", "C", "D", "E", "F", "G"],  # x_cat1
        ["A", "B", "C", "D", "E", "F", "G"],  # x_cat2
    ],
    "x_int": [
        (1, 20),  # x_int
    ],
    "x_con": [
        (-1.0, 1.0),  # x_con1
        (-1.0, 1.0),  # x_con2
        (-1.0, 1.0),  # x_con3
        (-1.0, 1.0),  # x_con4
        (-1.0, 1.0),  # x_con5
        (-1.0, 1.0),  # x_con6
        (-1.0, 1.0),  # x_con7
    ],
}

# s(x1^cat, x2^cat): rows = x2, cols = x1
_S_TABLE = np.array(
    [
        [0.40, 0.66, 0.91, 1.06, 0.89, 0.61, 0.37],  # x2 = A
        [0.56, 0.81, 1.07, 1.26, 1.09, 0.78, 0.52],  # x2 = B
        [0.71, 0.97, 1.22, 1.43, 1.24, 0.94, 0.69],  # x2 = C
        [0.91, 1.17, 1.43, 1.67, 1.47, 1.14, 0.85],  # x2 = D
        [1.12, 1.37, 1.63, 1.92, 1.70, 1.31, 1.03],  # x2 = E
        [1.36, 1.61, 1.94, 2.23, 1.96, 1.57, 1.26],  # x2 = F
        [1.61, 1.93, 2.24, 2.53, 2.22, 1.82, 1.51],  # x2 = G
    ],
    dtype=float,
)

_CAT_TO_IDX = {c: i for i, c in enumerate(BOUNDS["x_cat"][0])}


def cat_17(X):
    """
    Ishigami-based mixed-variable problem (with categorical table s).

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

    # integer in {1..20}
    x_int = X_int[:, 0].astype(float)

    X_con = X_con.astype(float)
    x1 = X_con[:, 0]
    x2 = X_con[:, 1]
    x3 = X_con[:, 2]
    x4 = X_con[:, 3]
    x5 = X_con[:, 4]
    x6 = X_con[:, 5]
    x7 = X_con[:, 6]

    # --- s(x1^cat, x2^cat) with rows=x2, cols=x1 ---
    i1 = np.vectorize(_CAT_TO_IDX.get)(cat1)
    i2 = np.vectorize(_CAT_TO_IDX.get)(cat2)
    s = _S_TABLE[i2, i1]

    # --- Objective (matches PyMoo/NOMAD) ---
    pi = np.pi

    sin_pi_x1 = np.sin(pi * x1)
    sin_pi_x2 = np.sin(pi * x2)

    bracket1 = sin_pi_x1 + 7.0 * (sin_pi_x2**2) + 0.1 * ((pi * x3) ** 4) * sin_pi_x1
    term1 = (s * s) * bracket1

    bracket2 = (x4 * x4 + x5 * x5 - 0.5) + (x_int / 20.0) * np.sin(4.0 * (x4 + x5))
    term2 = s * bracket2

    bracket3 = ((x6 * x6 + x7 * x7 - 1.0) ** 2) + (x_int / 20.0) * np.cos(4.0 * (x6 - x7))
    term3 = 0.3 * bracket3

    f = term1 + term2 + term3

    h = np.zeros(X.shape[0])
    g = [[] for _ in range(X.shape[0])]
    return f, h, g
