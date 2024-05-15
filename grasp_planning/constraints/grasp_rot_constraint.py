import numpy as np
import casadi as ca
import spatial_casadi as sc
from grasp_planning.constraints.constraint_template import Constraint

class GraspRotConstraint(Constraint):
    def __init__(self, robot, q_ca, waypoint_ID, param_grasp_ca, theta=0.0, tolerance=0.1) -> None:
        super().__init__() 
        # q = q_ca[:n_dof, waypoint_ID] -> joint space of the robot (decision variable)
        # manip_frame = q_ca[n_dof:, waypoint_ID] -> manipulation frame (decision variable)
        # manip frame_pos - T_W_Grasp_pos = 0
        self._robot = robot
        self.theta = theta
        self.tolerance = tolerance
        self.param_grasp_ca = param_grasp_ca
        
        R_W_EEF = self._robot.compute_fk_ca(q_ca[:,waypoint_ID])
        R_G_EEF = self.param_grasp_ca[:3,:3].T @ R_W_EEF[:3,:3]
        Rpy_G_EEF = sc.Rotation.from_matrix(R_G_EEF).as_euler("xyz")

        self.g = ca.vertcat(Rpy_G_EEF[0], Rpy_G_EEF[1], Rpy_G_EEF[2])
        self.g_lb = ca.vertcat(-self.tolerance, -self.theta, -self.tolerance)
        self.g_ub = ca.vertcat(self.tolerance, self.theta, self.tolerance)

        self.g_eval = ca.Function('g_grasp_rot', [q_ca, param_grasp_ca], [self.g])

    def get_constraint(self):
        return self.g, self.g_lb, self.g_ub

    def do_eval(self, q, T_W_Grasp):
        return self.g_eval(q, T_W_Grasp)
