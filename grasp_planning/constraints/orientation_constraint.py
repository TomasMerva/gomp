import numpy as np
import casadi as ca
import spatial_casadi as sc
from grasp_planning.constraints.constraint_template import Constraint

class OrientationConstraint(Constraint):
    def __init__(self, robot, q_ca, paramca_T_W_Ref, tolerance=0.0) -> None:
        super().__init__() 
        # q = q_ca[:n_dof, waypoint_ID] -> joint space of the robot (decision variable)
        # manip_frame = q_ca[n_dof:, waypoint_ID] -> manipulation frame (decision variable)
        # manip frame_pos - T_W_Grasp_pos = 0
        self._robot = robot
        self.tolerance = tolerance
        R_W_EEF = self._robot.compute_fk_ca(q_ca)[:3, :3]
        R_W_Ref = paramca_T_W_Ref[:3,:3]

        
        R_Ref_EEF= R_W_Ref.T @ R_W_EEF
        self.g = ca.trace(R_Ref_EEF)

        self.g_lb = 2*np.cos(tolerance) + 1.0
        self.g_ub = ca.inf
        self.g_eval = ca.Function('g_grasp_pos', [q_ca, paramca_T_W_Ref], [self.g])
      
    def get_constraint(self):
        return self.g, self.g_lb, self.g_ub

    def do_eval(self, q, T_W_Ref):
        return self.g_eval(q, T_W_Ref)