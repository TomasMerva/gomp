import numpy as np
import casadi as ca
import spatial_casadi as sc
from grasp_planning.constraints.constraint_template import Constraint

class GraspPosConstraint(Constraint):
    def __init__(self, robot, q_ca, waypoint_ID, param_grasp_ca, tolerance=0.0) -> None:
        super().__init__() 
        # q = q_ca[:n_dof, waypoint_ID] -> joint space of the robot (decision variable)
        # manip_frame = q_ca[n_dof:, waypoint_ID] -> manipulation frame (decision variable)
        # manip frame_pos - T_W_Grasp_pos = 0
        self._robot = robot
        self.tolerance = tolerance
        Rpy_W_EEF = self._robot.compute_fk_rpy_ca(q_ca[:,waypoint_ID])
        
        self.g = Rpy_W_EEF[:3] - param_grasp_ca[:3,3]
        self.g_lb = np.zeros(3) - self.tolerance
        self.g_ub = np.zeros(3) + self.tolerance
        self.g_eval = ca.Function('g_grasp_pos', [q_ca, param_grasp_ca], [self.g])
      
    def get_constraint(self):
        return self.g, self.g_lb, self.g_ub

    def do_eval(self, q, T_W_Grasp):
        return self.g_eval(q, T_W_Grasp)
