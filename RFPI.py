import numpy as np

def negative_func(a):
   
    a_hat=(a-np.abs(a))/2


    return a_hat


def rfpi(
    A: np.ndarray,
    y: np.ndarray, 
    alpha=1, 
    max_iter=20, 
    tol=1e-6,
    verbose: bool = False,
):
    
    m,n = A.shape
    lam = 1 / (np.sqrt(m))
    # Initialize the reconstructed signal x
    x_hat = np.random.rand(n)
    x_hat = x_hat/np.linalg.norm(x_hat)

    for i in range(max_iter):
        # Calculate the one-sided gradient
        Y = np.diag(y)
        print(Y)
        delta_fl = -(Y @ A).T @ negative_func((Y @ A @ x_hat))
        
        #Gradient projection on sphere surface:
        fl_hat= delta_fl-np.dot(delta_fl,x_hat)

        # Update the reconstructed signal x
        x_new = x_hat - alpha * fl_hat
        u_l = np.sign(x_new) * np.maximum(np.abs(x_new) - (alpha/lam), 0)

        # Perform a normalization
        x_hat = u_l / np.linalg.norm(u_l, ord=2)

        # Compute the residuals
        r = y - np.sign(A @ x_hat)

        if verbose:
            print(
                f"Iteration: {i}, ||y - A @ x_hat||_2: {np.linalg.norm(r)}"
            )

        # Check for convergence and stop iterating if this is the case
        if np.linalg.norm(r) < tol:
            print(f"RFPI converged at iteration: {i}")
            break

        
    return x_hat