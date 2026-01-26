import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    # 1 categorical var: A..L
    "x_cat": [list("ABCDEFGHIJKL")],
    # no integer vars
    "x_int": [],
    # 2 continuous vars: x1 in [-3,3], x2 in [0.1,3]
    "x_con": [(-3.0, 3.0), (0.1, 3.0)],
}


def cat_cstrs_22(X):
    """
    Cat-cstrs-22: mixed-combined MAD.

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

    cat = X_cat[:, 0]
    x1 = X_con[:, 0].astype(float)
    x2 = X_con[:, 1].astype(float)

    # Objective f (piecewise by category)
    f = np.zeros(n, dtype=float)

    idx = (cat == "A")
    f[idx] = x1[idx] * x1[idx] + x2[idx] * x2[idx] + x1[idx] * x2[idx] - 1.0

    idx = (cat == "B")
    f[idx] = np.sin(x1[idx]) + 0.05 * np.abs(x2[idx])

    idx = (cat == "C")
    f[idx] = -np.cos(x2[idx]) + 0.03 * np.abs(x1[idx])

    idx = (cat == "D")
    f[idx] = 0.95 * (x1[idx] * x1[idx] + x2[idx] * x2[idx] + x1[idx] * x2[idx] - 1.0) + 0.10 * (x1[idx] - x2[idx])

    idx = (cat == "E")
    f[idx] = np.sin(x1[idx]) + 0.08 * (x1[idx] - 1.0) ** 2

    idx = (cat == "F")
    f[idx] = -np.cos(x2[idx]) + 0.06 * (x2[idx] - 1.0) ** 2

    idx = (cat == "G")
    f[idx] = -np.exp(x1[idx] - x2[idx])

    idx = (cat == "H")
    f[idx] = np.sinh(x1[idx] - 1.0) - 1.0 + 0.03 * np.abs(x2[idx] - 1.0)

    idx = (cat == "I")
    f[idx] = -np.log(x2[idx]) - 1.0 + 0.04 * np.abs(x1[idx])

    idx = (cat == "J")
    f[idx] = -0.92 * np.exp(x1[idx] - x2[idx]) - 0.05 * (x1[idx] - x2[idx])

    idx = (cat == "K")
    f[idx] = 1.05 * (np.sinh(x1[idx] - 1.0) - 1.0) + 0.02 * (x1[idx] - 1.0) ** 2

    idx = (cat == "L")
    f[idx] = -np.log(x2[idx]) - 1.0 + 0.06 * (x2[idx] - 0.5) ** 2

    # Constraints
    g1 = np.zeros(n, dtype=float)

    # {A,B,C}
    idx = np.isin(cat, ["A", "B", "C"])
    g1[idx] = 0.5 - (x1[idx] * x1[idx] + x2[idx] * x2[idx])

    # {D,E,F}
    idx = np.isin(cat, ["D", "E", "F"])
    g1[idx] = 3.0 * x1[idx] + x2[idx] + 2.5

    # {G,H,I}
    idx = np.isin(cat, ["G", "H", "I"])
    g1[idx] = x2[idx] - 0.05 * x1[idx] - 0.5

    # {J,K,L}
    idx = np.isin(cat, ["J", "K", "L"])
    g1[idx] = 0.9 * x1[idx] - x2[idx] + 1.0

    # global constraints
    g2 = (x1 - 1.0) ** 2 + (x2 - 1.0) ** 2 - 0.5
    g3 = x1 + x2 - 1.0

    G = np.stack([g1, g2, g3], axis=1)
    for j in range(n):
        g[j] = G[j, :].tolist()

    h = np.array([np.sum(np.maximum(0.0, np.array(gj, dtype=float)) ** 2) for gj in g], dtype=float)
    return f, h, g
