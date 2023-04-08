import numpy as np
import cvxpy as cvx


def convex(
    A: np.ndarray,
    y: np.ndarray,
):
    n = A.shape[1]
    x_hat = np.zeros(n)

    vx = cvx.Variable(n)
    # Minimize the L1 norm
    objective = cvx.Minimize(cvx.norm(vx, 1))
    # Define the constraint that y=sign(Ax) is satisfied
    constraints = [cvx.multiply(A @ vx, y) >= 1]
    # constraints = [cvx.sign(A @ vx) == y, cvx.norm1(A @ vx) == 200]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve()
    #result = prob.solve(solver=cvx.ECOS, verbose=True)
    x_hat1 = np.array(vx.value).squeeze()
    # Perform a normalization
    x_hat = x_hat1 / np.linalg.norm(x_hat1, ord=2)

    return x_hat
