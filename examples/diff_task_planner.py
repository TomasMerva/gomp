import numpy as np
import casadi as ca
import spatial_casadi as sc
import os
from grasp_planning.solver.robot_model import RobotKinematicModel
from grasp_planning.cost.costs import SquaredAccCost

class TGOMP():
    def __init__(self, n_waypoints, urdf, root_link, end_link):
        self._robot_model = RobotKinematicModel(urdf, root_link, end_link)
        self.n_dofs = self._robot_model.n_dofs
        self.manipulation_frame_dim = 6
        self.x_dim = self.n_dofs + self.manipulation_frame_dim
        self.n_waypoints = n_waypoints
        self.l_joint_limits, self.u_joint_limits = None, None

        # Optimization
        self.x = ca.SX.sym("x", self.x_dim, self.n_waypoints)
        self._objective = SquaredAccCost(self.n_waypoints, self.x_dim, manip_frame=True)

    def set_init_guess(self, q):
        self.x_init = np.tile(q, self.n_waypoints)

    def set_boundary_conditions(self, q_start, q_end=None):
        if self.l_joint_limits is None or self.u_joint_limits is None:
            # Reset joint limits
            self.l_joint_limits = np.full((self.x_dim, self.n_waypoints), -100.0)
            self.u_joint_limits = np.full((self.x_dim, self.n_waypoints), 100.0)

            # get joint limits
            joint_limits = self._robot_model.get_joint_pos_limits()
            for t in range(self.n_waypoints):
                self.l_joint_limits[:self.n_dofs, t] = joint_limits[:,0]
                self.u_joint_limits[:self.n_dofs, t] = joint_limits[:,1]

        # init boundary
        self.l_joint_limits[:,0] = q_start
        self.u_joint_limits[:,0] = q_start
    
    def setup_problem(self, verbose=False):
        x_ca_flatten = self.x.reshape((-1,1))

        options = {}
        options["ipopt.acceptable_tol"] = 1e-3
        if not verbose:
            options["ipopt.print_level"] = 0
            options["print_time"] = 0

        self.solver = ca.nlpsol('solver', 'ipopt', {'x': x_ca_flatten, 'f': self._objective.eval_cost(x_ca_flatten)}, options)


    def add_grasp_constraint(self, waypoint_ID):
        Rpy_W_EEF = self._robot_model.compute_fk_rpy_ca(self.x[:self.n_dofs, waypoint_ID])
        
        self.g_grasp = Rpy_W_EEF[:3] - self.x[self.n_dofs:(self.n_dofs+3), waypoint_ID]
        self.g_grasp_lb = np.zeros(3)
        self.g_grasp_ub = np.zeros(3)
        

if __name__=="__main__":
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    URDF_FILE = absolute_path + "/assets/dingo_kinova_gripper.urdf"

    n_waypoints = 3 
    planner = TGOMP(n_waypoints, URDF_FILE, "world", "arm_tool_frame")
    x_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.57, 1.57,  1.57, 1.57, 0.0570004, -0.0100005, 1.00325, 0, 0, 0], dtype=float)
    planner.set_init_guess(x_init)
    planner.set_boundary_conditions(x_init)

    planner.setup_problem(verbose=True)
