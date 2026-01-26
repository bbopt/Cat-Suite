import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    # 1 categorical var: A..R
    "x_cat": [list("ABCDEFGHIJKLMNOPQR")],
    # no integer vars
    "x_int": [],
    # 16 continuous vars x1..x16 in [0.1, 10]
    "x_con": [(0.1, 10.0)] * 16,
}


def cat_cstrs_21(X):
    """
    Cat-cstrs-21: mixed Dembo 7 (vectorized inputs).

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

    # categorical labels
    cat = X_cat[:, 0]

    # continuous x1..x16
    x = [X_con[:, i].astype(float) for i in range(16)]
    (x1, x2, x3, x4, x5, x6, x7, x8,
     x9, x10, x11, x12, x13, x14, x15, x16) = x

    # constants
    a = 1.262626
    b = -1.231060
    c = 0.034750
    d = 0.009750

    # f1 (vectorized)
    f1 = (
        a * (x12 + x13 + x14 + x15 + x16)
        + b * (x1 * x12 + x2 * x13 + x3 * x14 + x4 * x15 + x5 * x16)
    )

    # s_term (piecewise by category)
    s_term = np.zeros(n, dtype=float)

    # A..E
    idx = (cat == "A")
    s_term[idx] = c * x1[idx] / x6[idx] + 100.0 * d * x1[idx] - d * x1[idx] / x6[idx] - 1.0

    idx = (cat == "B")
    s_term[idx] = c * x2[idx] / x7[idx] + 100.0 * d * x2[idx] - d * x2[idx] / x7[idx] - 1.0

    idx = (cat == "C")
    s_term[idx] = c * x3[idx] / x8[idx] + 100.0 * d * x3[idx] - d * x3[idx] / x8[idx] - 1.0

    idx = (cat == "D")
    s_term[idx] = c * x4[idx] / x9[idx] + 100.0 * d * x4[idx] - d * x4[idx] / x9[idx] - 1.0

    idx = (cat == "E")
    s_term[idx] = c * x5[idx] / x10[idx] + 100.0 * d * x5[idx] - d * x5[idx] / x10[idx] - 1.0

    # F
    idx = (cat == "F")
    s_term[idx] = (
        c * x6[idx] / x7[idx]
        + (x1[idx] / x5[idx]) * x11[idx] * x12[idx]
        - (x6[idx] / x5[idx]) * x1[idx] * x2[idx]
        - 1.0
    )

    # G
    idx = (cat == "G")
    s_term[idx] = (
        x7[idx] / x8[idx]
        + 0.002 * (x7[idx] - x2[idx]) * x1[idx] * x8[idx] * x12[idx]
        - 0.002 * (x7[idx] - x2[idx]) * x5[idx]
        - 1.0
    )

    # H
    idx = (cat == "H")
    s_term[idx] = (
        x8[idx]
        + 0.002 * (x8[idx] - x2[idx]) * x5[idx] * x8[idx]
        + 0.002 * (x3[idx] - x9[idx]) * x14[idx]
        + x9[idx]
        - 1.0
    )

    # I
    idx = (cat == "I")
    s_term[idx] = (
        (x9[idx] / x3[idx])
        + (x4[idx] - x8[idx]) * (x15[idx] / (x3[idx] * x14[idx]))
        + 500.0 * (x10[idx] - x9[idx]) / (x3[idx] * x14[idx])
        - 1.0
    )

    # J
    idx = (cat == "J")
    s_term[idx] = (
        ((x6[idx] / x4[idx]) - 1.0) * (x16[idx] / x15[idx])
        + (x10[idx] / x4[idx])
        + 500.0 * (1.0 - (x10[idx] / x4[idx])) / x15[idx]
        - 1.0
    )

    # K
    idx = (cat == "K")
    s_term[idx] = 0.9 / x4[idx] + 0.002 * (1.0 - (x5[idx] / x4[idx])) * x16[idx] - 1.0

    # L..R simple ratios
    idx = (cat == "L")
    s_term[idx] = x11[idx] / x12[idx] - 1.0

    idx = (cat == "M")
    s_term[idx] = x4[idx] / x5[idx] - 1.0

    idx = (cat == "N")
    s_term[idx] = x3[idx] / x4[idx] - 1.0

    idx = (cat == "O")
    s_term[idx] = x2[idx] / x3[idx] - 1.0

    idx = (cat == "P")
    s_term[idx] = x1[idx] / x2[idx] - 1.0

    idx = (cat == "Q")
    s_term[idx] = x9[idx] / x10[idx] - 1.0

    idx = (cat == "R")
    s_term[idx] = x8[idx] / x9[idx] - 1.0

    # full objective
    f = f1 + 1.0e3 * s_term

    # constraints g_i(x) <= 0 (vectorized)
    g1 = 0.002 * (x11 - x12) - 1.0
    g2 = x4 / x5 - 1.05
    g3 = x3 / x4 - 1.05
    g4 = x8 / x9 - 1.05
    g5 = x6 / x7 - 1.10
    g6 = x13 - 0.8 * x14
    g7 = x2 + x16 - 8.0

    G = np.stack([g1, g2, g3, g4, g5, g6, g7], axis=1)
    for j in range(n):
        g[j] = G[j, :].tolist()

    h = np.array([np.sum(np.maximum(0.0, np.array(gj, dtype=float)) ** 2) for gj in g], dtype=float)
    return f, h, g
