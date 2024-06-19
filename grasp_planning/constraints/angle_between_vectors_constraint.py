import numpy as np
import casadi as ca
import spatial_casadi as sc
from grasp_planning.constraints.constraint_template import Constraint

class AngleBetweenVectorsConstraint(Constraint):
    def __init__(self, robot, q_ca, waypoint_ID, paramca_T_W_Grasp, theta=0.0, axis="x") -> None:
        super().__init__() 
        self._robot = robot
        self.theta = theta
        if axis == "x":
            self.axis = 0
        elif axis == "y":
            self.axis = 1
        elif axis == "z":
            self.axis = 2
        # Grasp
        x_G = paramca_T_W_Grasp[:3,self.axis] / ca.norm_2(paramca_T_W_Grasp[:3,self.axis])
        # X_EEF
        R_W_EEF = self._robot.compute_fk_ca(q_ca[:,waypoint_ID])
        x_EEF = R_W_EEF[:3,self.axis] / ca.norm_2(R_W_EEF[:3,self.axis])

        self.g = x_G.T @ np.eye(3) @ x_EEF

        self.g_lb = ca.cos(theta)
        self.g_ub = ca.cos(-theta)

        self.g_eval = ca.Function('g_grasp_anglevectors', [q_ca, paramca_T_W_Grasp], [self.g])

    def get_constraint(self):
        return self.g, self.g_lb, self.g_ub

    def do_eval(self, q, T_W_Grasp):
        return self.g_eval(q, T_W_Grasp), self.g_lb, self.g_ub
