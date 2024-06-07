import numpy as np
import casadi as ca
import spatial_casadi as sc
from grasp_planning.constraints.constraint_template import Constraint


class GraspManipRotConstraint(Constraint):
    def __init__(self, robot, q_ca, waypoint_ID, paramca_T_W_Grasp, theta=0.0) -> None:
        super().__init__() 
        self._robot = robot
        self.theta = theta
        self.tolerance = 0.01
        R_W_EEF = q_ca[self._robot.n_dofs:,waypoint_ID]
        # R_W_EEF = self._robot.compute_fk_ca(q_ca[:,waypoint_ID])
        x_G = paramca_T_W_Grasp[:3,0] / ca.norm_2(paramca_T_W_Grasp[:3,0])
   
        x_EEF = R_W_EEF[:3,0] / ca.norm_2(R_W_EEF[:3,0])

        self.g =  x_G.T @ np.eye(3) @ x_EEF
        self.g_lb = np.cos(self.theta)
        self.g_ub = np.cos(-self.theta)
   
        self.g_eval = ca.Function('g_grasp_manip_rot', [q_ca, paramca_T_W_Grasp], [self.g])

    def get_constraint(self):
        return self.g, self.g_lb, self.g_ub

    def do_eval(self, q, T_W_Grasp):
        return self.g_eval(q, T_W_Grasp), self.g_lb, self.g_ub
