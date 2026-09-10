import numpy as np
from _utils import split_into_components, check_bounds

# Define BOUNDS / METADATA
BOUNDS = {
    "x_cat": [
        ["A", "B", "C"],
        ["A", "B", "C"]
    ],
    "x_int": [
        (-10, 10),
        (-10, 10)
    ],
    "x_con": [
        (-10.0, 10.0),
        (-10.0, 10.0),
        (-10.0, 10.0)
    ]
}


def cat_cstrs_6(X):
    """
    Cat-cstrs-6 based on the G-09 problem.

    Parameters:
        X : ndarray of shape (n_samples, n_variables)

    Returns:
        f : ndarray of shape (n_samples,)
        h : ndarray of shape (n_samples,)
        g : list of lists — constraint values per sample
    """
    X = np.atleast_2d(X)
    X_cat, X_int, X_con = split_into_components(X, BOUNDS)
    check_bounds(X_cat, X_int, X_con, BOUNDS)

    cat1 = X_cat[:, 0]
    cat2 = X_cat[:, 1]

    i1 = X_int[:, 0]
    i2 = X_int[:, 1]

    x1, x2, x3 = X_con.T

    # Lookup dictionaries for p and s.
    # Keys are (cat1, cat2).
    table_p = {
        ("A", "A"): -5.0,
        ("B", "A"): 2.5,
        ("C", "A"): 7.5,

        ("A", "B"): -5.0,
        ("B", "B"): 3.5,
        ("C", "B"): 8.5,

        ("A", "C"): -4.0,
        ("B", "C"): 4.5,
        ("C", "C"): 10.0,
    }

    table_s = {
        ("A", "A"): -5.0,
        ("B", "A"): -5.0,
        ("C", "A"): -4.0,

        ("A", "B"): 2.5,
        ("B", "B"): 4.5,
        ("C", "B"): 3.0,

        ("A", "C"): 1.0,
        ("B", "C"): 6.5,
        ("C", "C"): 9.0,
    }

    # Vectorized table lookup
    keys = list(zip(cat1, cat2))
    p = np.array([table_p[k] for k in keys], dtype=float)
    s = np.array([table_s[k] for k in keys], dtype=float)

    # Objective function
    f = (
        (x1 - 10.0) ** 2
        + 5.0 * (x2 - 12.0) ** 2
        + x3 ** 4
        + 3.0 * (i1 - 11.0) ** 2
        + 10.0 * i2 ** 6
        + 7.0 * p ** 2
        + s ** 2
        - 4.0 * p * s
        - 10.0 * p
        - 8.0 * s
    )

    # Constraints
    g1 = (
        2.0 * x1 ** 2
        + 3.0 * x2 ** 4
        + x3
        + 4.0 * i1 ** 2
        + 5.0 * i2
    ) / 2.0 - 127.0

    g2 = (
        7.0 * x1
        + 3.0 * x2
        + 10.0 * x3 ** 2
        + i1
        - i2
    ) / 2.0 - 282.0

    g3 = (
        23.0 * x1
        + x2 ** 2
        + p
        - 8.0 * s
        - 196.0
    )

    g4 = (
        4.0 * x1 ** 2
        + x2 ** 2
        - 3.0 * x1 * x2
        + 2.0 * x3 ** 2
        + 5.0 * p
        - 11.0 * s
    )

    g = [
        [g1[k], g2[k], g3[k], g4[k]]
        for k in range(X.shape[0])
    ]

    h = np.array([
        sum(max(0.0, gj) ** 2 for gj in gk)
        for gk in g
    ])

    return f, h, g
