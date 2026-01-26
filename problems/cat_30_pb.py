import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    "x_cat": [
        ["A", "B", "C"],  # x_cat1
        ["A", "B", "C"],  # x_cat2
        ["A", "B", "C"],  # x_cat3
        ["A", "B", "C"],  # x_cat4
    ],
    "x_int": [
        (0, 10),  # x_int1
        (0, 10),  # x_int2
        (0, 10),  # x_int3
        (0, 10),  # x_int4
    ],
    "x_con": [
        (0.0, 10.0),  # x_con1
        (0.0, 10.0),  # x_con2
        (0.0, 10.0),  # x_con3
        (0.0, 10.0),  # x_con4
        (0.0, 10.0),  # x_con5
        (0.0, 10.0),  # x_con6
    ],
}


def _s1(x1c: np.ndarray, x2c: np.ndarray, cat: np.ndarray) -> np.ndarray:
    out = np.empty_like(x1c, dtype=float)

    m = (cat == "A")
    out[m] = 0.35 + 0.10 * np.log(1.0 + (x1c[m] - 3.0) ** 2 + (x2c[m] - 7.0) ** 2)

    m = (cat == "B")
    out[m] = 0.30 + 0.06 * np.abs(x1c[m] - 5.0) + 0.04 * np.abs(x2c[m] - 2.0)

    m = (cat == "C")
    out[m] = 0.55 + 0.80 / (1.0 + (x1c[m] - 8.0) ** 2 + (x2c[m] - 1.0) ** 2)

    return out


def _s2(x1c: np.ndarray, x2c: np.ndarray, cat: np.ndarray) -> np.ndarray:
    out = np.empty_like(x1c, dtype=float)

    m = (cat == "A")
    out[m] = 0.30 + 0.06 * np.abs(x1c[m] - 2.0) + 0.05 * np.abs(x2c[m] - 6.0)

    m = (cat == "B")
    out[m] = 0.55 + 0.85 / (1.0 + (x1c[m] - 1.0) ** 2 + (x2c[m] - 9.0) ** 2)

    m = (cat == "C")
    out[m] = 0.33 + 0.10 * np.log(1.0 + (x1c[m] - 6.0) ** 2 + (x2c[m] - 4.0) ** 2)

    return out


def _s3(x1c: np.ndarray, x2c: np.ndarray, cat: np.ndarray) -> np.ndarray:
    out = np.empty_like(x1c, dtype=float)

    m = (cat == "A")
    out[m] = 0.55 + 0.75 / (1.0 + (x1c[m] - 4.0) ** 2 + (x2c[m] - 4.0) ** 2)

    m = (cat == "B")
    out[m] = 0.34 + 0.10 * np.log(1.0 + (x1c[m] - 9.0) ** 2 + (x2c[m] - 2.0) ** 2)

    m = (cat == "C")
    out[m] = 0.28 + 0.07 * np.abs(x1c[m] - 7.0) + 0.03 * np.abs(x2c[m] - 5.0)

    return out


def _s4(x1c: np.ndarray, x2c: np.ndarray, cat: np.ndarray) -> np.ndarray:
    out = np.empty_like(x1c, dtype=float)

    m = (cat == "A")
    out[m] = 0.33 + 0.10 * np.log(1.0 + (x1c[m] - 2.0) ** 2 + (x2c[m] - 1.0) ** 2)

    m = (cat == "B")
    out[m] = 0.55 + 0.90 / (1.0 + (x1c[m] - 7.0) ** 2 + (x2c[m] - 8.0) ** 2)

    m = (cat == "C")
    out[m] = 0.29 + 0.06 * np.abs(x1c[m] - 4.0) + 0.05 * np.abs(x2c[m] - 7.0)

    return out


def cat_30(X):
    """
    Cat-30: Shekel-based mixed-variable problem.

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

    c1 = X_cat[:, 0]
    c2 = X_cat[:, 1]
    c3 = X_cat[:, 2]
    c4 = X_cat[:, 3]

    ints = X_int.astype(float)  # (n,4)
    i1, i2, i3, i4 = [ints[:, k] for k in range(4)]

    X_con = X_con.astype(float)
    x1c, x2c, x3c, x4c, x5c, x6c = [X_con[:, k] for k in range(6)]

    # --- s_i functions ---
    s1 = _s1(x1c, x2c, c1)
    s2 = _s2(x1c, x2c, c2)
    s3 = _s3(x1c, x2c, c3)
    s4 = _s4(x1c, x2c, c4)

    svals = np.stack([s1, s2, s3, s4], axis=1)  # (n,4)

    # ----- main inverse-sum -----
    base_abs = np.abs(x1c - x2c + 0.4 * x5c - 0.3 * x6c)  # (n,)

    # Build denominators for k=1..4 vectorized
    # denom_k = s_k + (x1-ik)^2 + (x2-ik)^2 + 0.08*(x5-0.7ik)^2 + 0.06*(x6-0.4ik)^2 + 0.05*base_abs
    denom = (
        svals
        + (x1c[:, None] - ints) ** 2
        + (x2c[:, None] - ints) ** 2
        + 0.08 * (x5c[:, None] - 0.7 * ints) ** 2
        + 0.06 * (x6c[:, None] - 0.4 * ints) ** 2
        + 0.05 * base_abs[:, None]
    )
    sum_inv = np.sum(1.0 / denom, axis=1)

    # ----- penalties using x1..x4 continuous with i1..i4 -----
    con_for_pen = np.stack([x1c, x2c, x3c, x4c], axis=1)  # (n,4)
    d = con_for_pen - ints
    pen_abs = np.sum(np.abs(d), axis=1)
    pen_sqrt = np.sum(np.sqrt(np.abs(d)), axis=1)

    # ----- cross penalties -----
    cross1 = 0.03 * np.abs(x5c - x6c)
    cross2 = 0.02 * np.abs(x1c * x2c - x5c * x6c)

    # ----- oscillatory term -----
    phase = (np.pi / 5.0) * (x5c + x6c)  # (n,)
    osc = np.sum((1.0 + 0.3 * svals) * np.cos(phase[:, None] + 0.4 * ints), axis=1)

    f = (
        sum_inv
        + 0.12 * pen_abs
        + 0.04 * pen_sqrt
        + cross1
        + cross2
        + 0.06 * osc
    )

    h = np.zeros(X.shape[0])
    g = [[] for _ in range(X.shape[0])]
    return f, h, g
