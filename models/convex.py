import numpy as np
import cvxpy as cvx


def convex(
    A: np.ndarray,
    y: np.ndarray,
):
    """
    Function performs the convex optimization algorithm.

    Parameters
    ----------
    A : np.ndarray
        Measurement matrix (m, n)
    y : np.ndarray
        Measurement results, sign(Ax) (m x 1)

    Returns
    -------
    np.ndarray
        Returns the best x_hat with a normalization
    """

    n = A.shape[1]
    x_hat = np.zeros(n)

    vx = cvx.Variable(n)
    # Minimize the L1 norm
    objective = cvx.Minimize(cvx.norm(vx, 1))
    # Define the constraint that y=sign(Ax) is satisfied
    constraints = [cvx.multiply(A @ vx, y) >= 1]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve()
    #Some different solvers could be used to solve the convex problem
    #result = prob.solve(solver=cvx.ECOS) 
    #Converts a NumPy array to a one-dimensional array bu removing any dimension of length 1
    x_hat1 = np.array(vx.value).squeeze()
    # Perform a normalization
    x_hat = x_hat1 / np.linalg.norm(x_hat1, ord=2)

    return x_hat
