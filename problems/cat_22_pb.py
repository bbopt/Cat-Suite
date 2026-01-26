import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    "x_cat": [
        [str(i) for i in range(1, 52)],  # "1"..."51"
    ],
    "x_int": [],
    "x_con": [
        (-10.0, 10.0),  # x_con1
        (-10.0, 10.0),  # x_con2
        (-10.0, 10.0),  # x_con3
        (-10.0, 10.0),  # x_con4
        (-10.0, 10.0),  # x_con5
        (-10.0, 10.0),  # x_con6
    ],
}

# "k" -> k (integer)
_CAT_TO_I = {c: int(c) for c in BOUNDS["x_cat"][0]}


def _s_from_i(i: np.ndarray) -> np.ndarray:
    """
    Vectorized version of EVD61Problem._s_from_i.
    i in {1,...,51}, shape (n_samples,)
    """
    i = np.asarray(i, dtype=float)
    odd = (np.asarray(i, dtype=int) % 2 == 1)
    return np.where(odd, 0.12 * (i - 1.0) + 0.20, -0.09 * (i - 1.0) + 0.50)


def cat_22(X):
    """
    Cat-22: EVD61-based mixed-variable problem.

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

    cat = X_cat[:, 0]
    i = np.vectorize(_CAT_TO_I.get)(cat)  # i in {1,...,51}

    X_con = X_con.astype(float)
    x1 = X_con[:, 0]
    x2 = X_con[:, 1]
    x3 = X_con[:, 2]
    x4 = X_con[:, 3]
    x5 = X_con[:, 4]
    x6 = X_con[:, 5]

    # --- s(x^cat) ---
    s = _s_from_i(i)
    abs_s = np.abs(s)

    # --- Build pieces (matches NOMAD) ---
    t = np.abs(x3 * s + x4)
    frac_t = t / (1.0 + t)

    denom = 1.0 + (x2 * s) ** 2

    # --- Objective (matches NOMAD exactly) ---
    term1 = (1.0 + 0.15 * s) * (x1 * x1)
    term2 = (1.0 + 0.10 * abs_s) * (x5 * x5)

    inner = 1.0 - 0.25 * frac_t
    gain = (2.0 - 0.25 * s) * (x1 / denom) * inner
    term3 = -gain

    term4 = -0.6 * (1.0 + 0.08 * s * s) * np.cos(0.6 * x6 + 1.1 * s)
    term5 = 0.02 * s * x2 * x6

    f = term1 + term2 + term3 + term4 + term5

    h = np.zeros(X.shape[0])
    g = [[] for _ in range(X.shape[0])]
    return f, h, g
