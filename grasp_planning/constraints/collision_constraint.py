import numpy as np
import casadi as ca
import spatial_casadi as sc
from grasp_planning.constraints.constraint_template import Constraint

class CollisionConstraint(Constraint):
    def __init__(self, robot, q_ca, waypoint_ID, param_obst_ca, link_name, r_link, r_obst, tolerance=0.0) -> None:
        super().__init__() 
        self._robot = robot
        self.tolerance = tolerance
        self.r_link = r_link
        self.r_obst = r_obst
        self.d_safety = tolerance
        
        T_W_link = self._robot.compute_fk_ca(q_ca, link_name)

        self.g = ca.norm_2(T_W_link[:3,3]-param_obst_ca[:3,3]) #obstacle_pos
        self.g_lb = self.r_link + self.r_obst + self.d_safety
        self.g_ub = float("inf")


    def get_constraint(self):
        return self.g, self.g_lb, self.g_ub

    def do_eval(self, q, T_W_Grasp):
        pass
        #return self.g_eval(q, T_W_Grasp)
