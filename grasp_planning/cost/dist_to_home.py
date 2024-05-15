import numpy as np

"""
Finite Differencing Matrix to produce sum of squared accelerations
ddq= A * q
ddq.T * ddq = ddq.T * (A.T * A) * ddq
ddq.T * ddq = ddq.T * P * ddq
"""

class DistToHome():
    def __init__(self, q_home, n_dofs):
        self._n_dofs = n_dofs 
        self.q_home = q_home
        self.I = np.eye(self._n_dofs)

    def eval_cost(self, q):
        return (q -self.q_home).T @ self.I @ (q - self.q_home)