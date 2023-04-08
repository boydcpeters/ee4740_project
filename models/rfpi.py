import numpy as np

from tqdm import tqdm


def negative_func(a):
    a_hat = (a - np.abs(a)) / 2.0

    return a_hat


def rfpi(
    A: np.ndarray,
    y: np.ndarray,
    delta=0.01,
    lam=10,
    max_iter=20,
    tol=1e-6,
    verbose: bool = False,
):
    m, n = A.shape

    # Initialize the reconstructed signal x
    x_hat = np.linalg.pinv(A) @ y
    # x_hat = np.random.rand(n)
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

        # # Check for convergence and stop iterating if this is the case
        # if np.linalg.norm(r) < tol:
        #     print(f"RFPI converged at iteration: {i}")
        #     break

    return x_hat


def rfpi_adap(
    A: np.ndarray,
    y: np.ndarray,
    delta=0.01,
    factor=1.1,
    threshold=1e-6,
    max_iter=100,
    tol=1e-6,
    verbose: bool = False,
):
    # Calculate the lowest initial lambda
    lam_init = 100 * delta

    # Set the lambda value to the initial lambda value
    lam = lam_init

    # Create an array to store previous x_hat
    x_hat_prev = np.zeros(A.shape[1], dtype=np.float64)

    i = 1
    while True:
        x_hat = rfpi(
            A, y, delta=delta, lam=lam, max_iter=max_iter, tol=tol, verbose=True
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
