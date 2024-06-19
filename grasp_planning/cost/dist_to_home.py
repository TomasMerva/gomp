import numpy as np


class DistToHome():
    def __init__(self, q_home, n_dofs):
        self._n_dofs = n_dofs 
        self.q_home = q_home
        self.I = np.eye(self._n_dofs)
        print(self.q_home.shape)

    def eval_cost(self, q):
        return (q -self.q_home).T @ self.I @ (q - self.q_home)