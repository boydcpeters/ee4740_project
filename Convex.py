import numpy as np
import cvxpy as cvx

def Con(
    A: np.ndarray,
    y: np.ndarray,
):

    n = A.shape[1]
    x_hat = np.zeros(n)

    vx = cvx.Variable(n)
    # Minimize the L1 norm
    objective = cvx.Minimize(cvx.norm(vx, 1))
    # Define the constraint that y=sign(Ax) is satisfied
    constraints = [cvx.multiply(A@vx,y) >=1]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(solver=cvx.ECOS, verbose=True)
    x_hat = np.array(vx.value).squeeze()

    return x_hat