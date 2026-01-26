import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    "x_cat": [
        ["A", "B", "C", "D"],          # x_cat1
        ["A", "B", "C", "D"],          # x_cat2
        ["A", "B", "C", "D", "E"],     # x_cat3
    ],
    "x_int": [
        (-10, 10),  # x_int1
        (-10, 10),  # x_int2
    ],
    "x_con": [
        (-2.0, 2.0),  # x_con1
        (-2.0, 2.0),  # x_con2
        (-2.0, 2.0),  # x_con3
        (-2.0, 2.0),  # x_con4
    ],
}

_CAT12_TO_IDX = {c: i for i, c in enumerate(BOUNDS["x_cat"][0])}
_CAT3_TO_IDX = {c: i for i, c in enumerate(BOUNDS["x_cat"][2])}

# s1(x1^cat, x2^cat): rows = x2, cols = x1
_S1_TABLE = np.array(
    [
        [0.80, 1.10, 0.95, 1.20],  # x2 = A
        [1.05, 0.85, 1.25, 0.90],  # x2 = B
        [0.92, 1.18, 1.00, 0.88],  # x2 = C
        [1.15, 0.98, 0.87, 1.30],  # x2 = D
    ],
    dtype=float,
)


def _s2(cat3_idx: np.ndarray, x3: np.ndarray) -> np.ndarray:
    """
    Vectorized version of CamelProblem._s2.
    cat3_idx shape: (n_samples,)
    x3 shape: (n_samples,)
    """
    cat3_idx = np.asarray(cat3_idx, dtype=int)
    x3 = np.asarray(x3, dtype=float)

    out = np.empty_like(x3, dtype=float)

    m = (cat3_idx == 0)  # A
    out[m] = np.abs(x3[m] - 0.5) + 0.05

    m = (cat3_idx == 1)  # B
    out[m] = 1.05 * np.abs(x3[m] - 0.5) + 0.05

    m = (cat3_idx == 2)  # C
    out[m] = 0.95 * np.abs(x3[m] - 0.5) + 0.06

    m = (cat3_idx == 3)  # D
    out[m] = 0.70 + 0.30 * np.exp(-0.5 * (x3[m] + 2.0))

    m = (cat3_idx == 4)  # E
    out[m] = 1.00 + 0.40 * np.exp(-0.8 * (x3[m] + 1.0))

    return out


def cat_20(X):
    """
    Cat-20: Camel-based mixed-variable problem.

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
    cat3 = X_cat[:, 2]

    # integers
    x_int1 = X_int[:, 0].astype(float)
    x_int2 = X_int[:, 1].astype(float)

    # continuous
    X_con = X_con.astype(float)
    x1 = X_con[:, 0]
    x2 = X_con[:, 1]
    x3 = X_con[:, 2]
    x4 = X_con[:, 3]

    # --- s1 table lookup: rows=x2, cols=x1 ---
    i1 = np.vectorize(_CAT12_TO_IDX.get)(cat1)
    i2 = np.vectorize(_CAT12_TO_IDX.get)(cat2)
    s1 = _S1_TABLE[i2, i1]

    # --- Integer scaling factor ---
    scale = 1.0 + 0.03 * (x_int1 - 5.0) + 0.02 * (x_int2 - 4.0)

    # --- Six-hump camel-like bracket ---
    x1_2 = x1 * x1
    x1_4 = x1_2 * x1_2
    x2_2 = x2 * x2

    camel = (
        (4.0 - 2.1 * x1_2 + (1.0 / 3.0) * x1_4) * x1_2
        + x1 * x2
        + (-4.0 + 4.0 * x2_2) * x2_2
    )
    term_camel = s1 * scale * camel

    # --- Quadratic regularization term ---
    term_quad = 0.2 * (x3 * x3 + (x4 - 1.0) * (x4 - 1.0))

    # --- Coupling squared term ---
    coupling = (x3 * x1 + x4 * x2)
    term_coupling = 0.05 * coupling * coupling

    # --- Mixed integer-continuous interaction ---
    term_mix = 0.01 * (x_int1 - 5.0) * (x_int2 - 4.0) * (x3 + x4)

    # --- s2(x3^cat, x3^continuous) ---
    c3 = np.vectorize(_CAT3_TO_IDX.get)(cat3)
    s2 = _s2(c3, x3)

    f = term_camel + term_quad + term_coupling + term_mix + 0.1 * s2

    h = np.zeros(X.shape[0])
    g = [[] for _ in range(X.shape[0])]
    return f, h, g
