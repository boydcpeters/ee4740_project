import numpy as np
from tqdm import tqdm


def negative_func(a):
    """
    Function preserves all the negative elements and sets all positive elements to 0.

    Parameters
    ----------
    a : np.ndarray
        Array over which the negative function should be taken

    Returns
    -------
    np.ndarray
        Array where all the positive values are set to zero.
    """
    idx = a > 0
    a[idx] = 0

    return a


def eta_k(a, k):
    """
    Returns an array with only the k highest absolute values and the rest of the
    values are set to zero.

    Hard threshold function.

    Parameters
    ----------
    a : np.ndarray
        Array of interest
    k : int
        Number of highest absolute values to include.

    Returns
    -------
    np.ndarray
        k-sparse approximation
    """

    # Retrieve the highest coefficient indices
    indices = np.argsort(np.abs(a))[::-1][:k]

    a_high = np.zeros(a.shape)
    a_high[indices] = a[indices]

    return a_high


def biht(
    A: np.ndarray,
    y: np.ndarray,
    k: int,
    max_iter: int = 10000,
    mode: str = "l1",
    verbose: bool = False,
):
    """
    Function performs the Binary Iterative Hard Thresholding (BIHT) algorithm.

    Parameters
    ----------
    A : np.ndarray
        Measurement matrix (m, n)
    y : np.ndarray
        Measurement results, sign(Ax) (m x 1)
    k : int
        Sparsity level
    max_iter : int, optional
        Maximum number of iterations, by default 3000.
    mode : str, optional
        Set the mode for the gradient calculation, by default "l1".
            - "l1": y is noiseless
            - "l2": y has noise
    verbose: bool, optional
        Indicates whether the iteration number and the residual error
        at that iteration should be printed out.

    Returns
    -------
    Returns the best x with sparsity level k.
    """

    if mode not in {"l1", "l2"}:
        raise ValueError("The mode should be either 'l1' or 'l2'.")

    m, n = A.shape

    # TODO: understand how this works for l2 norm, because the
    # gradient will always be zero right if x0 = np.zeros(n)??
    x_hat = np.zeros(n)

    # Calculate the gradient step size
    lam = 1 / (np.sqrt(m))

    for i in tqdm(range(max_iter)):
        # Calculate the one-sided gradient
        if mode == "l1":
            delta_fl = lam / 2 * A.T @ (y - np.sign(A @ x_hat))
        elif mode == "l2":
            Y = np.diag(y)
            delta_fl = lam / 2 * (Y @ A).T @ negative_func((Y @ A @ x_hat))

        # Compute and update
        temp = x_hat + delta_fl
        u_l = eta_k(temp, k)

        # Perform a normalization
        x_hat = u_l / np.linalg.norm(u_l, ord=2)

        # Compute the residuals
        r = y - np.sign(A @ x_hat)

        if verbose:
            print(
                f"Iteration: {i}, ||y - A @ x_hat||_2: {np.linalg.norm(r)}"
            )  # , "x_hat: ", x_hat)

        # Check for convergence and stop iterating if this is the case
        if np.linalg.norm(r) < 1e-6:
            print(f"BIHT converged at iteration: {i}")
            break

    return x_hat
