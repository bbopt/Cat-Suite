import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    "x_cat": [["A", "B", "C"]] * 4,      # x_cat1..x_cat4
    "x_int": [(0, 10)] * 4,              # x_int1..x_int4
    "x_con": [(0.0, 10.0)] * 6,          # x_con1..x_con6
}


def cat_cstrs_29(X):
    """
    Cat-cstrs-29: constrained Shekel-like mixed problem.

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

    # cats (strings)
    c1, c2, c3, c4 = (X_cat[:, k] for k in range(4))

    # ints
    i1 = X_int[:, 0].astype(int)
    i2 = X_int[:, 1].astype(int)
    i3 = X_int[:, 2].astype(int)
    i4 = X_int[:, 3].astype(int)

    # cons
    x1 = X_con[:, 0].astype(float)
    x2 = X_con[:, 1].astype(float)
    x3 = X_con[:, 2].astype(float)
    x4 = X_con[:, 3].astype(float)
    x5 = X_con[:, 4].astype(float)
    x6 = X_con[:, 5].astype(float)

    # --- s_i(cat_i, x1, x2) ---
    def s1(cat, u1, u2):
        if cat == "A":
            return 0.35 + 0.10 * np.log(1.0 + (u1 - 3.0) ** 2 + (u2 - 7.0) ** 2)
        if cat == "B":
            return 0.30 + 0.06 * np.abs(u1 - 5.0) + 0.04 * np.abs(u2 - 2.0)
        # "C"
        return 0.55 + 0.80 / (1.0 + (u1 - 8.0) ** 2 + (u2 - 1.0) ** 2)

    def s2(cat, u1, u2):
        if cat == "A":
            return 0.30 + 0.06 * np.abs(u1 - 2.0) + 0.05 * np.abs(u2 - 6.0)
        if cat == "B":
            return 0.55 + 0.85 / (1.0 + (u1 - 1.0) ** 2 + (u2 - 9.0) ** 2)
        # "C"
        return 0.33 + 0.10 * np.log(1.0 + (u1 - 6.0) ** 2 + (u2 - 4.0) ** 2)

    def s3(cat, u1, u2):
        if cat == "A":
            return 0.55 + 0.75 / (1.0 + (u1 - 4.0) ** 2 + (u2 - 4.0) ** 2)
        if cat == "B":
            return 0.34 + 0.10 * np.log(1.0 + (u1 - 9.0) ** 2 + (u2 - 2.0) ** 2)
        # "C"
        return 0.28 + 0.07 * np.abs(u1 - 7.0) + 0.03 * np.abs(u2 - 5.0)

    def s4(cat, u1, u2):
        if cat == "A":
            return 0.33 + 0.10 * np.log(1.0 + (u1 - 2.0) ** 2 + (u2 - 1.0) ** 2)
        if cat == "B":
            return 0.55 + 0.90 / (1.0 + (u1 - 7.0) ** 2 + (u2 - 8.0) ** 2)
        # "C"
        return 0.29 + 0.06 * np.abs(u1 - 4.0) + 0.05 * np.abs(u2 - 7.0)

    S1 = np.array([s1(c1[j], x1[j], x2[j]) for j in range(n)], dtype=float)
    S2 = np.array([s2(c2[j], x1[j], x2[j]) for j in range(n)], dtype=float)
    S3 = np.array([s3(c3[j], x1[j], x2[j]) for j in range(n)], dtype=float)
    S4 = np.array([s4(c4[j], x1[j], x2[j]) for j in range(n)], dtype=float)

    abs_mix = np.abs(x1 - x2 + 0.4 * x5 - 0.3 * x6)

    def inv_term(Si, ii):
        ii = ii.astype(float)
        d = (
            Si
            + (x1 - ii) ** 2
            + (x2 - ii) ** 2
            + 0.08 * (x5 - 0.7 * ii) ** 2
            + 0.06 * (x6 - 0.4 * ii) ** 2
            + 0.05 * abs_mix
        )
        return 1.0 / d

    sum_inv = (
        inv_term(S1, i1)
        + inv_term(S2, i2)
        + inv_term(S3, i3)
        + inv_term(S4, i4)
    )

    pen_abs = (
        np.abs(x1 - i1.astype(float))
        + np.abs(x2 - i2.astype(float))
        + np.abs(x3 - i3.astype(float))
        + np.abs(x4 - i4.astype(float))
    )

    pen_sqrt = (
        np.sqrt(np.abs(x1 - i1.astype(float)))
        + np.sqrt(np.abs(x2 - i2.astype(float)))
        + np.sqrt(np.abs(x3 - i3.astype(float)))
        + np.sqrt(np.abs(x4 - i4.astype(float)))
    )

    trig = (
        (1.0 + 0.3 * S1) * np.cos((np.pi / 5.0) * (x5 + x6) + 0.4 * i1.astype(float))
        + (1.0 + 0.3 * S2) * np.cos((np.pi / 5.0) * (x5 + x6) + 0.4 * i2.astype(float))
        + (1.0 + 0.3 * S3) * np.cos((np.pi / 5.0) * (x5 + x6) + 0.4 * i3.astype(float))
        + (1.0 + 0.3 * S4) * np.cos((np.pi / 5.0) * (x5 + x6) + 0.4 * i4.astype(float))
    )

    f = (
        sum_inv
        + 0.12 * pen_abs
        + 0.04 * pen_sqrt
        + 0.03 * np.abs(x5 - x6)
        + 0.02 * np.abs(x1 * x2 - x5 * x6)
        + 0.06 * trig
    ).astype(float)

    # Constraints (<= 0)
    g1 = (S1 + S2) * (S3 + S4) + (i1 + i2 + i3 + i4).astype(float) / 12.0 - 2.60
    g2 = (x3 - x4) ** 2 + (x5 - x6) ** 2 + 0.20 * np.abs(x1 - i1.astype(float)) + 0.20 * np.abs(x2 - i2.astype(float)) - 3.20
    g3 = (x1 - 4.0) * (x2 - 6.0) + 0.30 * np.abs((i1 - i2).astype(float)) + 0.20 * np.abs(S2 - S4) - 2.50

    Gmat = np.stack([g1, g2, g3], axis=1).astype(float)
    for j in range(n):
        g[j] = Gmat[j, :].tolist()

    h = np.array([np.sum(np.maximum(0.0, np.array(gj, dtype=float)) ** 2) for gj in g], dtype=float)
    return f, h, g
