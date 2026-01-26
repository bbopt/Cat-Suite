import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    # x_cat in {0,...,12} encoding A..M
    "x_cat": [list(range(13))],
    # 10 integer vars in [-20,20]
    "x_int": [(-20, 20)] * 10,
    # 10 continuous vars in [-20,20]
    "x_con": [(-20.0, 20.0)] * 10,
}


def cat_cstrs_23(X):
    """
    Cat-cstrs-23: modified Wong-3 (constrained).

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

    cat = X_cat[:, 0].astype(int)

    # integers (xi1..xi10)
    xi = [X_int[:, i].astype(float) for i in range(10)]
    xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8, xi9, xi10 = xi

    # continuous (xc1..xc10)
    xc = [X_con[:, i].astype(float) for i in range(10)]
    xc1, xc2, xc3, xc4, xc5, xc6, xc7, xc8, xc9, xc10 = xc

    # ---- s_inner(cat, xc, xi) vectorized by cat ----
    s_inner = np.zeros(n, dtype=float)

    idx = (cat == 0)  # A
    s_inner[idx] = 3.0 * (xc1[idx] - 2.0) ** 2 + 4.0 * (xc2[idx] - 3.0) ** 2 + 2.0 * (xc3[idx] ** 2) - 7.0 * xc4[idx] - 120.0

    idx = (cat == 1)  # B
    s_inner[idx] = 5.0 * (xc1[idx] ** 2) + 8.0 * xc2[idx] + (xc3[idx] - 6.0) ** 2 - 2.0 * xc4[idx] - 40.0

    idx = (cat == 2)  # C
    s_inner[idx] = 0.5 * (xc1[idx] - 8.0) ** 2 + 2.0 * (xc2[idx] - 4.0) ** 2 + 3.0 * (xc5[idx] ** 2) - xc6[idx] - 30.0

    idx = (cat == 3)  # D
    s_inner[idx] = (xc1[idx] ** 2) + 2.0 * (xc2[idx] - 2.0) ** 2 - 2.0 * xc1[idx] * xc2[idx] + 14.0 * xc5[idx] - 6.0 * xc6[idx]

    idx = (cat == 4)  # E
    s_inner[idx] = -3.0 * xc1[idx] + 6.0 * xc2[idx] + 12.0 * (xc8[idx] - 8.0) ** 2 - 7.0 * xc10[idx]

    idx = (cat == 5)  # F
    s_inner[idx] = (xc1[idx] ** 2) + 5.0 * xc1[idx] - 8.0 * xc2[idx] - 28.0

    idx = (cat == 6)  # G
    s_inner[idx] = 4.0 * xc1[idx] + 9.0 * xc2[idx] + 5.0 * (xi3[idx] ** 2) - 9.0 * xi4[idx] - 87.0

    idx = (cat == 7)  # H
    s_inner[idx] = 3.0 * xc1[idx] + 4.0 * xc2[idx] + 3.0 * (xi3[idx] - 6.0) ** 2 - 14.0 * xi4[idx] - 10.0

    idx = (cat == 8)  # I
    s_inner[idx] = 14.0 * (xi2[idx] ** 2) + 35.0 * xi5[idx] - 79.0 * xi6[idx] - 92.0

    idx = (cat == 9)  # J
    s_inner[idx] = 15.0 * (xi5[idx] ** 2) + 11.0 * xi5[idx] - 61.0 * xi6[idx] - 54.0

    idx = (cat == 10)  # K
    s_inner[idx] = 5.0 * (xc1[idx] ** 2) + 2.0 * xc2[idx] + 9.0 * (xi7[idx] ** 4) - xi8[idx] - 68.0

    idx = (cat == 11)  # L
    s_inner[idx] = (xc1[idx] ** 2) - xc9[idx] + 19.0 * xi9[idx] - 20.0 * xi10[idx] + 19.0

    idx = (cat == 12)  # M
    s_inner[idx] = 12.0 * (xc2[idx] ** 2) + (xc9[idx] ** 2) - 30.0 * xi10[idx]

    s = 10.0 * s_inner

    # ---- f1 (vectorized) ----
    f1 = (
        (xc1 ** 2) + (xc2 ** 2) + (xc1 * xc2)
        - 14.0 * xc1 - 16.0 * xc2
        + (xc3 - 10.0) ** 2
        + 4.0 * (xc4 - 5.0) ** 2
        + (xc5 - 3.0) ** 2
        + 2.0 * (xc6 - 1.0) ** 2
        + 5.0 * (xc7 ** 2)
        + 7.0 * (xc8 - 11.0) ** 2
        + 2.0 * (xc9 - 10.0) ** 2
        + (xc10 - 7.0) ** 2
        + (xi1 - 9.0) ** 2
        + 10.0 * (xi2 - 1.0) ** 2
        + 5.0 * (xi3 - 7.0) ** 2
        + 4.0 * (xi4 - 14.0) ** 2
        + 27.0 * (xi5 - 1.0) ** 2
        + (xi6 ** 2)
        + (xi7 - 2.0) ** 2
        + 13.0 * (xi8 - 2.0) ** 2
        + (xi9 - 3.0) ** 2
        + (xi10 ** 2)
        + 95.0
        + 0.30 * np.sin(xc1)
        + 0.25 * np.cos(0.5 * xc4)
        + 0.20 * np.sin(0.3 * xc9)
        + 0.08 * np.abs(xc5 - 3.0) * np.abs(xc6 - 1.0)
        + 0.05 * np.abs(xi3 - 7.0) * np.abs(xi4 - 14.0)
        + 0.03 * (xc7 - 0.5 * xc8) * (xi1 - 9.0)
        + 0.02 * (xc3 - 10.0) * (xi2 - 1.0)
    )

    f = f1 + s

    # ---- constraints ----
    g1 = (
        4.0 * xc1 + 5.0 * xc2 - 3.0 * xc7 + 9.0 * xc8 - 105.0
        + 0.15 * (xc1 ** 2) + 0.05 * np.abs(xi1)
    )
    g2 = (
        10.0 * xc1 - 8.0 * xc2 - 17.0 * xc7 + 2.0 * xc8
        + 0.10 * (xc2 - 1.0) ** 2 + 0.03 * (xi2 - 1.0) ** 2
    )
    g3 = (
        -8.0 * xc1 + 2.0 * xc2 + 5.0 * xc9 - 2.0 * xc10 - 12.0
        + 0.06 * (xc9 - 8.0) ** 2 + 0.02 * np.abs(xi3 - 6.0)
    )
    g4 = (
        xc1 + xc2 + 4.0 * xi1 - 21.0 * xi2
        + 0.08 * (xc3 - 10.0) ** 2 + 0.05 * np.abs(xi4 - 14.0)
    )

    G = np.stack([g1, g2, g3, g4], axis=1)
    for j in range(n):
        g[j] = G[j, :].tolist()

    h = np.array([np.sum(np.maximum(0.0, np.array(gj, dtype=float)) ** 2) for gj in g], dtype=float)
    return f, h, g
