import numpy as np

"""
Finite Differencing Matrix to produce sum of squared accelerations
ddq= A * q
ddq.T * ddq = ddq.T * (A.T * A) * ddq
ddq.T * ddq = ddq.T * P * ddq
"""

class SquaredAccCost():
    def __init__(self, num_waypoints, num_dof, manip_frame = False):
        self._n_waypoints = num_waypoints
        self._n_dof = num_dof 
        self._manip_frame = manip_frame
        self.P = self.create_P_matrix(self._n_waypoints, self._n_dof)

    def create_P_matrix(self, num_waypoints, num_dof):
        if self._manip_frame:
            FD_matrix = np.zeros((num_waypoints, num_waypoints), dtype=float)
            FD_INDICES = [1., -2., 1.]
            for i in range(1, num_waypoints-1):
                FD_matrix[i, (i-1):(i-1)+3] = FD_INDICES
            FD_matrix[0,0]=1
            FD_matrix[-1,-1] = 1
            R = FD_matrix.T@FD_matrix
            I = np.eye(num_dof)
            I[-6:] = 0.0 # dont want to penalize q_dd for grasp frame
            return np.kron(R, I)
        else:
            FD_matrix = np.zeros((num_waypoints, num_waypoints), dtype=float)
            FD_INDICES = [1., -2., 1.]
            for i in range(1, num_waypoints-1):
                FD_matrix[i, (i-1):(i-1)+3] = FD_INDICES
            FD_matrix[0,0]=1
            FD_matrix[-1,-1] = 1
            R = FD_matrix.T @ FD_matrix
            return np.kron(R, np.eye(num_dof))

    def eval_cost(self, q):
        return q.T @ self.P @ q