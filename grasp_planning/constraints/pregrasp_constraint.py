import numpy as np
import casadi as ca
import spatial_casadi as sc
from grasp_planning.constraints.constraint_template import Constraint

class PreGraspConstraint(Constraint):
    def __init__(self, robot, q_ca, waypoint_ID, grasp_waypoint_ID, offset= -0.1, tolerance=0.0) -> None:
        super().__init__() 
        self._robot = robot

        # T_W_EEF = self._robot.compute_fk_ca(q_ca[: ,waypoint_ID])
        # T_W_grasp = self._robot.compute_fk_ca(q_ca[:, grasp_waypoint_ID])
        T_W_EEF = q_ca[self._robot.n_dofs: ,waypoint_ID]
        T_W_grasp = q_ca[self._robot.n_dofs:, grasp_waypoint_ID]

        # T_Grasp_PreG = np.eye(4)
        # T_Grasp_PreG[2,3] = offset
        # T_W_Preg_ref = T_W_grasp@T_Grasp_PreG
        # T_W_EEF[2] += offset
        g_pos = T_W_EEF[:3] - T_W_grasp[:3] 
        g_pos_lb = np.zeros(3) - 0.01
        g_pos_ub = np.zeros(3) + 0.01
        g_pos_lb[2] = -offset
        g_pos_ub[2] = -offset

        R_EEF_rpy = sc.Rotation.from_euler("xyz", T_W_EEF[3:]).as_matrix() #convert rotation part of FK into rpy
        R_Grasp_rpy = sc.Rotation.from_euler("xyz", T_W_grasp[3:]).as_matrix() #convert rotation part of FK into rpy


        T_Preg_EEF = R_Grasp_rpy.T @ R_EEF_rpy
        g_rot = ca.trace(T_Preg_EEF)
        g_rot_lb = 2*np.cos(tolerance) + 1.0
        g_rot_ub = ca.inf

        self.g = ca.vertcat(g_pos, g_rot)
        self.g_lb = ca.vertcat(g_pos_lb, g_rot_lb)
        self.g_ub = ca.vertcat(g_pos_ub, g_rot_ub)

        self.g_eval = ca.Function('g_pregrasp', [q_ca], [self.g])

    def get_constraint(self):
        return self.g, self.g_lb, self.g_ub

    def do_eval(self, q, T_W_Grasp):
        return self.g_eval(q, T_W_Grasp), self.g_lb, self.g_ub
