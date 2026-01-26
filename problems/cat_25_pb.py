import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    "x_cat": [
        [chr(ord("A") + i) for i in range(18)],  # A..R
    ],
    "x_int": [(-50, 50)] * 10,      # x_int1..x_int10
    "x_con": [(-50.0, 50.0)] * 10,  # x_con1..x_con10
}


def cat_25(X):
    """
    Cat-25: Wong3-based mixed-variable problem.

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

    I = X_int.astype(float)  # (n,10)
    C = X_con.astype(float)  # (n,10)

    # helpers (0-based column access)
    I1, I2, I3, I4, I5, I6, I7, I8, I9, I10 = [I[:, k] for k in range(10)]
    C1, C2, C3, C4, C5, C6, C7, C8, C9, C10 = [C[:, k] for k in range(10)]

    # --- g(x^int, x^con) ---
    g_val = np.zeros(X.shape[0], dtype=float)

    g_val += np.abs(I1) + np.abs(C1) + I1 * C1 - 14.0 * I1 - 16.0 * C1
    g_val += (I2 - 10.0) ** 2 + 4.0 * (C2 - 5.0) ** 2

    g_val += np.abs(I3 - 3.0) + 2.0 * np.abs(C3 - 1.0)
    g_val += 5.0 * (I4 ** 2) + 7.0 * (C4 - 11.0) ** 2

    g_val += 2.0 * np.abs(I5 - 10.0) + np.abs(C5 - 7.0)
    g_val += (I6 - 9.0) ** 2 + 10.0 * (C6 - 1.0) ** 2

    g_val += 5.0 * np.abs(I7 - 7.0) + 4.0 * np.abs(C7 - 14.0)
    g_val += 27.0 * (I8 - 1.0) ** 2 + (C8 ** 4)

    g_val += np.abs(I9 - 2.0) + 13.0 * np.abs(C9 - 2.0)
    g_val += (I10 - 3.0) ** 2 + (C10 ** 2)

    g_val += 95.0

    # --- s(x^cat, x^int, x^con) casework (vectorized with masks) ---
    s_case = np.zeros(X.shape[0], dtype=float)

    # A: 0 already

    m = (cat == "B")
    s_case[m] = (
        3.0 * (I1[m] - 2.0) ** 2
        + 4.0 * (C1[m] - 3.0) ** 2
        + 2.0 * (I2[m] ** 2)
        - 7.0 * C2[m]
        - 120.0
    )

    m = (cat == "C")
    s_case[m] = (
        5.0 * (I1[m] ** 2)
        + 8.0 * C1[m]
        + (I2[m] - 6.0) ** 2
        - 2.0 * C2[m]
        - 40.0
    )

    m = (cat == "D")
    s_case[m] = (
        0.5 * (I1[m] - 8.0) ** 2
        + 2.0 * (C1[m] - 4.0) ** 2
        + 3.0 * (I3[m] ** 2)
        - C3[m]
        - 30.0
    )

    m = (cat == "E")
    s_case[m] = (
        (I1[m] ** 2)
        + 2.0 * (C1[m] - 2.0) ** 2
        - 2.0 * I1[m] * C1[m]
        + 14.0 * I3[m]
        - 6.0 * C3[m]
    )

    m = (cat == "F")
    s_case[m] = (
        4.0 * (I2[m] ** 2)
        + 5.0 * C2[m]
        - 3.0 * I4[m]
        + 9.0 * C4[m]
        - 105.0
    )

    m = (cat == "G")
    s_case[m] = 10.0 * I1[m] - 8.0 * C1[m] - 17.0 * I4[m] + 2.0 * C4[m]

    m = (cat == "H")
    s_case[m] = (
        -3.0 * I1[m]
        + 6.0 * C1[m]
        + 12.0 * (I5[m] - 8.0) ** 2
        - 7.0 * C5[m]
    )

    m = (cat == "I")
    s_case[m] = -8.0 * I1[m] + 2.0 * C1[m] + 5.0 * I5[m] - 2.0 * C5[m] - 12.0

    m = (cat == "J")
    s_case[m] = I1[m] + C1[m] + 4.0 * I6[m] - 21.0 * C6[m]

    m = (cat == "K")
    s_case[m] = (I1[m] ** 2) + 5.0 * I6[m] - 8.0 * C6[m] - 28.0

    m = (cat == "L")
    s_case[m] = 4.0 * I1[m] + 9.0 * C1[m] + 5.0 * (I7[m] ** 2) - 9.0 * C7[m] - 87.0

    m = (cat == "M")
    s_case[m] = 3.0 * I1[m] + 4.0 * C1[m] + 3.0 * (I7[m] - 6.0) ** 2 - 14.0 * C7[m] - 10.0

    m = (cat == "N")
    s_case[m] = 14.0 * (I1[m] ** 2) + 35.0 * I8[m] - 79.0 * C8[m] - 92.0

    m = (cat == "O")
    s_case[m] = 15.0 * (C1[m] ** 2) + 11.0 * I8[m] - 61.0 * C8[m] - 54.0

    m = (cat == "P")
    s_case[m] = 5.0 * (I1[m] ** 2) + 2.0 * C1[m] + 9.0 * (I9[m] ** 4) - C9[m] - 68.0

    m = (cat == "Q")
    s_case[m] = (I1[m] ** 2) - C1[m] + 19.0 * I10[m] - 20.0 * C10[m] + 19.0

    m = (cat == "R")
    s_case[m] = 7.0 * (I1[m] ** 2) + 5.0 * (C1[m] ** 2) + (I10[m] ** 2) - 30.0 * C10[m]

    s = 10.0 * s_case
    f = g_val + s

    h = np.zeros(X.shape[0])
    g = [[] for _ in range(X.shape[0])]
    return f, h, g
