import numpy as np
import casadi as ca
import spatial_casadi as sc
from grasp_planning.constraints.constraint_template import Constraint


class ManipulationConstraint(Constraint):
    def __init__(self, robot, q_ca, waypoint_ID, n_dofs) -> None:
        super().__init__() 
        # q = q_ca[:n_dof, waypoint_ID] -> joint space of the robot (decision variable)
        # manip_frame = q_ca[n_dof:, waypoint_ID] -> manipulation frame (decision variable)
        # FK(q)_rpy - manip frame = 0
        self._robot = robot
        
        self.g = self._robot.compute_fk_rpy_ca(q_ca[:n_dofs, waypoint_ID]) - q_ca[n_dofs:, waypoint_ID] 
        self.g_lb = np.zeros(6)
        self.g_ub = np.zeros(6)
        self.g_eval = ca.Function('g_manip', [q_ca], [self.g])

        
    def get_constraint(self):
        return self.g, self.g_lb, self.g_ub
    
    def get_limits(self):
        return self.g_lb, self.g_ub
    
    def set_limits(self, lb, ub):
        self.g_lb = lb
        self.g_ub = ub
    
    def do_eval(self, q):
        return self.g_eval(q)
