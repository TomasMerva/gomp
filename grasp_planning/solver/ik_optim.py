import numpy as np
import casadi as ca
import spatial_casadi as sc
from scipy.spatial.transform import Rotation as R
from grasp_planning.solver.robot_model import RobotKinematicModel
from grasp_planning.cost.costs import *
from grasp_planning.constraints.constraints import *

class IK_OPTIM():
    def __init__(self, urdf, root_link='world', end_link='tool_frame') -> None:
        # Init variables
        self.T_W_Ref = None
        
        # Kinematics
        self._robot_model = RobotKinematicModel(urdf, root_link, end_link)
        self.n_dofs = self._robot_model.n_dofs
        self.x_dim = self.n_dofs

        # Optimization
        self.x = ca.SX.sym("x", self.x_dim, 1)
        self.x_init = None
        
        self._objective = None
        self.l_joint_limits, self.u_joint_limits = None, None
        self.g_list = []

        self.param_T_W_Ref = ca.SX.sym("param_grasp", 4, 4)
        self.param_obst_ca = ca.SX.sym("param_obst", 4, 4)
        self.param_optim_ca = ca.vertcat()
        self.param_ca_list = [self.param_T_W_Ref]
        self._collision_flag = False

    def set_home_config(self, q):
        self.q_home = q

    def set_init_guess(self, q):
        self.x_init = q

    def set_boundary_conditions(self):
        if self.l_joint_limits is None or self.u_joint_limits is None:
            # Reset joint limits
            self.l_joint_limits = np.full((self.x_dim, 1), -100.0)
            self.u_joint_limits = np.full((self.x_dim, 1), 100.0)

            # get joint limits
            joint_limits = self._robot_model.get_joint_pos_limits()
            self.l_joint_limits[:, 0] = joint_limits[:,0]
            self.u_joint_limits[:, 0] = joint_limits[:,1]

    

    def setup_problem(self, verbose=False):
        # assert self.T_Obj_Grasp != None, "Grasp DOF is not set. Set additional degree of freedom around object."
        # assert self.T_W_Obj != None, "Object pose is not set."
        # 1. create decision variable
        
        # assert self.x_init != None, "Initial guess is missing. Set up one using `set_init_guess(q)`."

        # Define objective function
        cost = DistToHome(q_home=self.q_home,
                          n_dofs=self.x_dim)
        objective = cost.eval_cost(self.x)

        g, self.g_lb, self.g_ub = ca.vertcat(), ca.vertcat(), ca.vertcat()
        for g_term in self.g_list:
            gt, g_lb, g_ub = g_term.get_constraint()
            g = ca.vertcat(g, gt)
            self.g_lb = ca.vertcat(self.g_lb, g_lb)
            self.g_ub = ca.vertcat(self.g_ub, g_ub)

        if self._collision_flag:
            self.param_ca_list.append(self.param_obst_ca)


        self.param_optim_ca = ca.vertcat(self.param_ca_list[0])
        for i in range(1, len(self.param_ca_list)):
            self.param_optim_ca = ca.vertcat(self.param_optim_ca, self.param_ca_list[i])

        options = {}
        options["ipopt.acceptable_tol"] = 1e-3
        if not verbose:
            options["ipopt.print_level"] = 0
            options["print_time"] = 0

        self.solver = ca.nlpsol('solver', 'ipopt', {'x': self.x, 'f': objective, 'g': g, 'p': self.param_optim_ca}, options)
    

        
    def solve(self):
        result = self.solver(x0=self.x_init,
                             lbg=self.g_lb,
                             ubg=self.g_ub,
                             lbx=self.l_joint_limits.reshape((-1,1), order='F'),
                             ubx=self.u_joint_limits.reshape((-1,1), order='F'),
                             p=self.params_optim_num)

        success_flag = self.solver.stats()["success"]
        success_msg = self.solver.stats()["return_status"]
        return  result['x'], success_flag

    def add_position_constraint(self, tolerance=0.0):
        self.g_list.append(PositionConstraint(robot=self._robot_model,
                                              q_ca=self.x,
                                              paramca_T_W_Ref=self.param_T_W_Ref,
                                              tolerance=tolerance))

    def add_orientation_constraint(self, tolerance=0.0):
        self.g_list.append(OrientationConstraint(robot=self._robot_model,
                                                 q_ca=self.x,
                                                 paramca_T_W_Ref=self.param_T_W_Ref,
                                                 tolerance=tolerance))

    

    def add_collision_constraint(self, child_link, r_link=0.5, r_obst=0.2, tolerance=0.01):
        self._collision_flag = True
        self.g_list.append(CollisionConstraint(robot=self._robot_model,
                                                q_ca=self.x,
                                                waypoint_ID=0,
                                                param_obst_ca=self.param_obst_ca,
                                                link_name=child_link,
                                                r_link= r_link,
                                                r_obst= r_obst,
                                                tolerance=tolerance))

    def update_constraints_params(self, T_W_Ref, T_W_Obst=None):
        self.T_W_Ref = T_W_Ref
        if self._collision_flag:
            self.params_optim_num = ca.vertcat(self.T_W_Ref, T_W_Obst)
        else:
            self.params_optim_num = self.T_W_Ref

