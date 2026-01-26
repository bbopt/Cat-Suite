import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    "x_cat": [
        [chr(ord("A") + i) for i in range(10)],  # A..J
    ],
    "x_int": [
        (-10, 10),  # x_int1
        (-10, 10),  # x_int2
        (-10, 10),  # x_int3 (present but unused in f)
    ],
    "x_con": [
        (0.0, 1.0),   # x_con1
        (-1.0, 1.0),  # x_con2
        (-1.0, 1.0),  # x_con3
        (-1.0, 1.0),  # x_con4
        (-1.0, 1.0),  # x_con5
    ],
}

_B_MAP = {
    "A": 0.6, "B": 1.1, "C": 2.7, "D": 3.5, "E": 4.3,
    "F": 4.0, "G": 3.4, "H": 2.3, "I": 1.0, "J": 0.4
}


def cat_26(X):
    """
    Roustant-based mixed-variable problem.

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

    # integers (x_int3 exists but is unused, keep it for shape consistency)
    x1_int = X_int[:, 0].astype(float)
    x2_int = X_int[:, 1].astype(float)

    # continuous
    X_con = X_con.astype(float)
    x1 = X_con[:, 0]
    x2 = X_con[:, 1]
    x3 = X_con[:, 2]
    x4 = X_con[:, 3]
    x5 = X_con[:, 4]

    # delta = round((x1_int + x2_int)/10)
    ratio = (x1_int + x2_int) / 10.0
    delta = np.round(ratio).astype(int).astype(float)

    # b(x^cat)
    b = np.array([_B_MAP.get(c, 0.0) for c in cat], dtype=float)

    u = b + delta
    pi = np.pi

    f = np.zeros(X.shape[0], dtype=float)

    # group 1: {"A","B","C","D"}
    m1 = np.isin(cat, ["A", "B", "C", "D"])
    if np.any(m1):
        termA = (x1[m1] + 0.01 * (x1[m1] - 0.5) ** 2)
        termB = (u[m1] / 10.0)
        termC = (1.0 + 0.1 * x2[m1] + 0.05 * x3[m1])
        termD = 0.02 * (x4[m1] * x4[m1] + x5[m1] * x5[m1])
        f[m1] = termA * termB * termC + termD

    # group 2: {"E","F","G"}
    m2 = np.isin(cat, ["E", "F", "G"])
    if np.any(m2):
        shift = (u[m2] - 4.0) / 20.0
        c = np.cos(2.0 * pi * (x1[m2] + shift))
        f[m2] = (
            0.9 * c * np.exp(-x1[m2]) * (1.0 + 0.1 * x2[m2])
            + 0.03 * x3[m2]
            + 0.02 * (x4[m2] * x4[m2] + x5[m2])
        )

    # group 3: {"H","I","J"}
    m3 = np.isin(cat, ["H", "I", "J"])
    if np.any(m3):
        shift = (u[m3] - 7.0) / 20.0
        c = np.cos(2.0 * pi * (x1[m3] + shift))
        f[m3] = (
            -0.7 * c * np.exp(-x1[m3]) * (1.0 + 0.1 * x2[m3] - 0.05 * x3[m3])
            + 0.02 * x4[m3] * x5[m3]
        )

    h = np.zeros(X.shape[0])
    g = [[] for _ in range(X.shape[0])]
    return f, h, g
