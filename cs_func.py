import numpy as np


def create_A(m, n, seed=None):
    # Create the random number generator to sample from the standard normal
    rng = np.random.default_rng(seed=seed)

    # Draw the entries of A independently from the standard Gaussian distribution
    A = rng.standard_normal(size=(m, n))

    # Normalize the columns to have a unit l2 norm
    A = A / np.linalg.norm(A, ord=2, axis=0)

    return A


def calc_y(A, x):
    return np.sign(A @ x)
