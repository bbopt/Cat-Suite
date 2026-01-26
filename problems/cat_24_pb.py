import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    "x_cat": [
        [str(i) for i in range(1, 22)],  # "1"..."21"
    ],
    "x_int": [
        (-5, 4),   # x_int1
        (-6, 8),   # x_int2
    ],
    "x_con": [
        (-25.0, 25.0),  # x_con1
        (-25.0, 25.0),  # x_con2
        (-25.0, 25.0),  # x_con3
    ],
}

# "k" -> k (integer in {1,...,21})
_CAT_TO_K = {c: int(c) for c in BOUNDS["x_cat"][0]}


def _s(k: np.ndarray, x3: np.ndarray) -> np.ndarray:
    """
    Vectorized version of OET5Problem._s.
    k shape: (n_samples,) in {1,...,21}
    x3 shape: (n_samples,)
    """
    k = np.asarray(k, dtype=float)
    x3 = np.asarray(x3, dtype=float)

    z = (x3 + 25.0) / 50.0

    out = np.empty_like(x3, dtype=float)

    m = (k <= 10.0)
    out[m] = 0.1 + 0.02 * z[m] + 0.01 * (k[m] - 5.0) * z[m]

    t = (x3[~m] + 25.0) / 10.0 + 0.3 * (k[~m] - 11.0)
    out[~m] = 0.2 + 0.08 * t * t

    return out


def cat_24(X):
    """
    Cat-24: OET5-based mixed-variable problem.

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
    k = np.vectorize(_CAT_TO_K.get)(cat)  # (n,) in {1,...,21}

    # integers
    x1_int = X_int[:, 0].astype(float)
    x2_int = X_int[:, 1].astype(float)

    # continuous
    X_con = X_con.astype(float)
    x1 = X_con[:, 0]
    x2 = X_con[:, 1]
    x3 = X_con[:, 2]

    # --- Compute s(x^cat, x3) ---
    s = _s(k, x3)

    # --- Factor A ---
    abs_x1 = np.abs(x1)
    abs_x2int = np.abs(x2_int)

    prod = x2_int * abs_x1
    sat_prod = prod / (1.0 + abs_x2int * abs_x1)

    sqrt_s = np.sqrt(s)
    sat_sqrt = sqrt_s / (1.0 + sqrt_s)

    factorA = sat_prod - sat_sqrt

    # --- Factor B ---
    num_inside = 0.5 * x1 * s * s + x2 * s * x1_int
    den_inside = x1 * s * s + x2 * s * np.abs(x1_int)

    num2 = num_inside * num_inside
    den2 = 1.0 + den_inside * den_inside

    factorB = 0.03 + (num2 / den2)
    factorB += 0.03 * (1.0 + 0.5 * np.cos(2.0 * np.pi * x3 + 0.7 * s))

    # --- Factor C ---
    u = 2.0 * np.pi * x3 + 3.0 * s
    factorC = 0.15 * np.cos(u) + 1.0

    f = factorA * factorB * factorC

    h = np.zeros(X.shape[0])
    g = [[] for _ in range(X.shape[0])]
    return f, h, g
