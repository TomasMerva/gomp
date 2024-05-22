import numpy as np
import casadi as ca
import spatial_casadi as sc
from grasp_planning.constraints.constraint_template import Constraint

class ZheightConstraint(Constraint):
    def __init__(self, robot, q_ca, waypoint_ID, z_height, tolerance=0.0) -> None:
        super().__init__() 
        self._robot = robot
        self.tolerance = tolerance
        T_W_EEF = self._robot.compute_fk_ca(q_ca[:,waypoint_ID])
        
        self.g = T_W_EEF[2,3]
        self.g_lb = z_height - self.tolerance
        self.g_ub = float("inf")
        self.g_eval = ca.Function('g_z_coord', [q_ca], [self.g])
      
    def get_constraint(self):
        return self.g, self.g_lb, self.g_ub

    def do_eval(self, q):
        return self.g_eval(q)
