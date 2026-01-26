import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    "x_cat": [list("ABCDEFGHIJ"), list("ABCDEFGHIJ")],  # x1_cat, x2_cat
    "x_int": [(-6, 18), (-6, 18)],                      # i1, i2
    "x_con": [(0.7, 1.5), (0.5, 1.4), (0.6, 1.6), (0.7, 1.5)],  # x1..x4
}


def cat_cstrs_24(X):
    """
    Cat-cstrs-24: Welded beam (constrained).

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

    # --- unpack ---
    x1_cat = X_cat[:, 0]
    x2_cat = X_cat[:, 1]

    i1 = X_int[:, 0].astype(float)
    i2 = X_int[:, 1].astype(float)

    x1 = X_con[:, 0].astype(float)
    x2 = X_con[:, 1].astype(float)
    x3 = X_con[:, 2].astype(float)
    x4 = X_con[:, 3].astype(float)

    # --- s-table: rows = x2_cat, cols = x1_cat ---
    cats = list("ABCDEFGHIJ")
    cat_to_idx = {c: k for k, c in enumerate(cats)}
    c1 = np.array([cat_to_idx[c] for c in x1_cat], dtype=int)  # columns
    c2 = np.array([cat_to_idx[c] for c in x2_cat], dtype=int)  # rows

    S = np.array([
        [0.40, 0.44, 0.50, 0.60, 0.74, 0.86, 0.92, 0.84, 0.70, 0.54],  # x2=A
        [0.62, 0.70, 0.80, 0.94, 1.12, 1.28, 1.36, 1.22, 1.02, 0.78],  # x2=B
        [0.82, 0.94, 1.08, 1.26, 1.50, 1.70, 1.78, 1.58, 1.30, 0.98],  # x2=C
        [0.98, 1.12, 1.30, 1.54, 1.84, 2.05, 2.08, 1.84, 1.50, 1.12],  # x2=D
        [1.08, 1.26, 1.48, 1.76, 2.12, 2.28, 2.20, 1.94, 1.58, 1.18],  # x2=E
        [1.02, 1.18, 1.38, 1.62, 1.96, 2.14, 2.16, 1.90, 1.54, 1.14],  # x2=F
        [0.90, 1.02, 1.18, 1.38, 1.66, 1.82, 1.90, 1.74, 1.42, 1.06],  # x2=G
        [0.74, 0.84, 0.96, 1.12, 1.34, 1.46, 1.56, 1.50, 1.24, 0.94],  # x2=H
        [0.56, 0.62, 0.70, 0.82, 0.98, 1.06, 1.14, 1.12, 1.00, 0.78],  # x2=I
        [0.36, 0.40, 0.46, 0.54, 0.66, 0.72, 0.78, 0.80, 0.72, 0.56],  # x2=J
    ], dtype=float)

    s = S[c2, c1].astype(float)

    # --- constants ---
    L = 14.0

    # --- helper definitions (vectorized) ---
    h_ = 0.18 + 0.22 * x1 + 0.006 * (i1 + 6.0)
    ell = 2.5 + 4.8 * x2 + 0.08 * (i2 + 6.0)
    t = 5.0 + 4.5 * x3 + 0.10 * i1
    b = 0.20 + 0.30 * x4
    P = 2600.0 + 1600.0 * x2 + 60.0 * i2

    # Material moduli
    E = 30.0e6 * (0.55 + 0.35 * s)
    G = 12.0e6 * (0.60 + 0.30 * s)

    # Shear stress components
    tau_p = P / (np.sqrt(2.0) * h_ * ell)
    M = P * (L + ell / 2.0)
    R = np.sqrt((ell * ell) / 4.0 + ((h_ + t) / 2.0) ** 2)
    J = 2.0 * (np.sqrt(2.0) * h_ * ell) * ((ell * ell) / 12.0 + ((h_ + t) ** 2) / 4.0)
    tau_pp = (M * R) / J

    tau = np.sqrt(tau_p * tau_p + 2.0 * tau_p * tau_pp * (ell / (2.0 * R)) + tau_pp * tau_pp)

    # Normal stress and deflection
    sigma = (6.0 * P * L) / (b * (t ** 2))
    delta = (4.0 * P * (L ** 3)) / (E * b * (t ** 3))

    # Buckling load Pc
    Pc = (
        (4.013 * E * np.sqrt((t * t * (b ** 6)) / 36.0) / (L ** 2))
        * (1.0 - (t / (2.0 * L)) * np.sqrt(E / (4.0 * G)))
    )

    # Objective
    base_cost = 1.10471 * (h_ ** 2) * ell + 0.04811 * t * b * (L + ell)
    f = (
        (1.35 - 0.25 * s) * base_cost
        + 10.0 * s * np.abs((i1 - i2) / 24.0)
        + 0.08 * np.abs(s - 1.35)
    )

    # Constraints (<= 0)
    g1 = tau - 13600.0 * s
    g2 = sigma - 30000.0 * s
    g3 = h_ - b
    g4 = delta - 0.25
    g5 = P - Pc

    Gmat = np.stack([g1, g2, g3, g4, g5], axis=1)
    for j in range(n):
        g[j] = Gmat[j, :].tolist()

    h = np.array([np.sum(np.maximum(0.0, np.array(gj, dtype=float)) ** 2) for gj in g], dtype=float)
    return f, h, g
