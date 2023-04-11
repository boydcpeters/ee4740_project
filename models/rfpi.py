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
    
    a_hat = (a - np.abs(a)) / 2

    return a_hat


def rfpi(
    A: np.ndarray,
    y: np.ndarray,
    delta=0.1,
    lam=10000,
    max_iter=1000,
    verbose: bool = False,
):
    """
    Function performs the Renormalized fixed point iteration (RFPI) algorithm.

    Parameters
    ----------
    A : np.ndarray
        Measurement matrix (m, n)

    y : np.ndarray
        Measurement results, sign(Ax) (m x 1)

    delta : int
        The descent step size

    lam: int
        The gradient step size

    max_iter : int, optional
        Maximum number of iterations, by default 1000.
    
    verbose: bool, optional
        Indicates whether the iteration number and the residual error
        at that iteration should be printed out.

    Returns
    -------
    np.ndarray
        Returns the best x_hat with normalization
    """
    #Set the gradient step size
    m, n = A.shape

    # Initialize the reconstructed signal x
    x_hat = np.random.rand(n)
    x_hat = x_hat / np.linalg.norm(x_hat)

    Y = np.diag(y)
    for i in tqdm(range(max_iter)) if verbose else range(max_iter):
        # Calculate the one-sided gradient
        delta_fl = ((Y @ A).T) @ negative_func((Y @ A @ x_hat))

        # Gradient projection on sphere surface:
        fl_hat = delta_fl - (delta_fl @ x_hat) * x_hat

        # Update the reconstructed signal x
        x_new = x_hat - delta * fl_hat
        u_l = np.sign(x_new) * np.maximum((np.abs(x_new) - delta / lam), np.zeros(n))

        # Perform a normalization
        x_hat = u_l / np.linalg.norm(u_l, ord=2)

        # Compute the residuals
        r = y - np.sign(A @ x_hat)

        if verbose:
            print(f"Iteration: {i}, ||y - A @ x_hat||_2: {np.linalg.norm(r)}")

    return x_hat


def rfpi_adap(
    A: np.ndarray,
    y: np.ndarray,
    delta=0.01,
    factor=1.1,
    max_iter=1000,
    verbose: bool = False,
):
    """
    Function performs the Renormalized fixed point iteration (RFPI) algorithm.

    Parameters
    ----------
    A : np.ndarray
        Measurement matrix (m, n)

    y : np.ndarray
        Measurement results, sign(Ax) (m x 1)

    delta : int
        The descent step size

    factor : int
        Updata the lam value

    max_iter : int, optional
        Maximum number of iterations, by default 1000.
    
    verbose: bool, optional
        Indicates whether the iteration number and the residual error
        at that iteration should be printed out.

    Returns
    -------
    np.ndarray
        Returns the best x_hat based on the best lam
    """

    # Calculate the lowest initial lambda
    lam_init = 100 * delta

    # Set the lambda value to the initial lambda value
    lam = lam_init

    # Create an array to store previous x_hat
    x_hat_prev = np.zeros(A.shape[1], dtype=np.float64)
    # Set the coverage threshold
    threshold=1e-6

    i = 1
    while True:
        x_hat = rfpi(
            A, y, delta=delta, lam=lam, max_iter=max_iter, verbose=True
        )

        diff = np.linalg.norm((x_hat - x_hat_prev))

        if diff < threshold:
            break
        else:
            x_hat_prev = np.copy(x_hat)

        lam = factor * lam

        print(
            f"Iteration: {i}  lambda: {lam}   diff: {diff}", "\r", end=""
        ) if verbose else None
        i = i + 1

    return x_hat