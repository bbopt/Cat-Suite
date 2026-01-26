import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    "x_cat": [
        [str(i) for i in range(1, 62)],  # "1"..."61"
    ],
    "x_int": [],
    "x_con": [
        (-1.0, 1.0),  # x_con1
        (-1.0, 1.0),  # x_con2
        (-1.0, 1.0),  # x_con3
        (-1.0, 1.0),  # x_con4
    ],
}

# s(x^cat) lookup table: index 0 -> "1", ..., index 60 -> "61"
_S_TABLE = np.array(
    [
        1.00, 1.26, 1.54, 1.84, 2.16, 2.50, 2.86,
        1.12, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40,
        3.00, 3.26, 3.54, 3.84, 4.16, 4.50, 4.86, 5.24, 5.64, 6.06, 6.50, 6.96, 7.44, 7.94,
        6.76, 7.20, 7.66,
        15.00, 14.85, 14.70, 14.55,
        12.00, 11.85, 11.70, 11.55, 11.40, 11.25, 11.10, 10.95, 10.80, 10.65, 10.50, 10.35, 10.20, 10.05,
        12.15, 11.05, 9.95, 8.85, 7.75, 6.65, 5.55, 4.45, 3.35, 2.25, 1.15, 1.00
    ],
    dtype=float,
)

_CAT_TO_IDX = {c: i for i, c in enumerate(BOUNDS["x_cat"][0])}

_EPS = 1e-2
_DELTA = 1e-6


def cat_20(X):
    """
    Cat-20: Gamma-based mixed-variable problem.

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
    X_con = X_con.astype(float)

    x1 = X_con[:, 0]
    x2 = X_con[:, 1]
    x3 = X_con[:, 2]
    x4 = X_con[:, 3]

    # --- s(x^cat) lookup ---
    idx = np.vectorize(_CAT_TO_IDX.get)(cat)
    s = _S_TABLE[idx]  # (n_samples,)

    # --- Stable objective (vectorized) ---
    eps = _EPS
    delta = _DELTA

    z = x3 * s + x4
    inner = s + x2 + (z / (z * z + eps * eps))
    base = np.sqrt(inner * inner + delta * delta)     # strictly positive
    exponent = s + 0.5
    ratio = base / (s + 1.0)                          # should be > 0

    # Compute power term in log-space for stability:
    # power_term = (ratio ** exponent) = exp(exponent * log(ratio))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        theta = exponent * np.log(ratio)
    theta = np.clip(theta, -700.0, 700.0)
    power_term = np.exp(theta)

    f = x1 * power_term - 1.0

    # Match PyMoo guard: any non-finite or invalid ratio/exponent -> huge penalty
    bad = (
        ~np.isfinite(ratio)
        | (ratio <= 0.0)
        | ~np.isfinite(exponent)
        | ~np.isfinite(f)
    )
    f = np.where(bad, 1e20, f)

    h = np.zeros(X.shape[0])
    g = [[] for _ in range(X.shape[0])]
    return f, h, g
