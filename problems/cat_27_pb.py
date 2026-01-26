import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    "x_cat": [
        [chr(ord("A") + i) for i in range(10)],  # x_cat1: A..J
        [chr(ord("A") + i) for i in range(10)],  # x_cat2: A..J
    ],
    "x_int": [
        (-50, 50),  # x_int1
    ],
    "x_con": [
        (-10.0, 10.0),  # x_con1
        (-10.0, 10.0),  # x_con2
        (-10.0, 10.0),  # x_con3
        (-10.0, 10.0),  # x_con4
    ],
}

# Categories A..J -> 0..9
_CAT_TO_IDX = {c: i for i, c in enumerate(BOUNDS["x_cat"][0])}

# s-table: rows = x2^{cat}, cols = x1^{cat}
_S_TABLE = np.array(
    [
        [0.783, 0.391, 0.196, 0.098, 0.049, 0.033, 0.025, 0.020, 0.016, 0.014],
        [0.779, 0.389, 0.195, 0.097, 0.049, 0.033, 0.024, 0.020, 0.016, 0.014],
        [0.694, 0.347, 0.174, 0.087, 0.043, 0.029, 0.022, 0.017, 0.014, 0.012],
        [0.640, 0.320, 0.160, 0.080, 0.040, 0.027, 0.020, 0.016, 0.013, 0.011],
        [0.338, 0.169, 0.084, 0.042, 0.021, 0.014, 0.011, 0.008, 0.007, 0.006],
        [0.251, 0.125, 0.063, 0.031, 0.016, 0.011, 0.008, 0.006, 0.005, 0.005],
        [0.182, 0.091, 0.046, 0.023, 0.011, 0.008, 0.006, 0.005, 0.004, 0.003],
        [0.137, 0.068, 0.034, 0.017, 0.009, 0.006, 0.004, 0.003, 0.003, 0.002],
        [0.129, 0.065, 0.032, 0.016, 0.008, 0.005, 0.004, 0.003, 0.003, 0.002],
        [0.094, 0.047, 0.024, 0.012, 0.006, 0.004, 0.003, 0.002, 0.002, 0.002],
    ],
    dtype=float,
)


def cat_27(X):
    """
    Kowalik-Osborne-based mixed-variable problem.

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

    x_int1 = X_int[:, 0].astype(float)

    X_con = X_con.astype(float)
    x1 = X_con[:, 0]
    x2 = X_con[:, 1]
    x3 = X_con[:, 2]
    x4 = X_con[:, 3]

    # --- s(x_cat1, x_cat2): rows=x_cat2, cols=x_cat1 ---
    i1 = np.vectorize(_CAT_TO_IDX.get)(cat1)
    i2 = np.vectorize(_CAT_TO_IDX.get)(cat2)
    s = _S_TABLE[i2, i1]

    s2 = s * s
    num = x1 * (s2 + x2 * s)
    den = (s2 + x3 * s + x4)

    f = (
        (num / den) - s
        + 0.05 * np.abs(x2 - x3 + 0.1 * x_int1)
        + 0.03 * np.abs(x1 * x4 - 0.05 * x_int1)
        + 0.02 * np.abs(x3 + 0.02 * x_int1)
    )

    h = np.zeros(X.shape[0])
    g = [[] for _ in range(X.shape[0])]
    return f, h, g
