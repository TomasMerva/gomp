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

        self.param_ca_dict = dict()


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
        # Read Constraints
        g, self.g_lb, self.g_ub = ca.vertcat(), ca.vertcat(), ca.vertcat()
        for g_term in self.g_list:
            gt, g_lb, g_ub = g_term.get_constraint()
            g = ca.vertcat(g, gt)
            self.g_lb = ca.vertcat(self.g_lb, g_lb)
            self.g_ub = ca.vertcat(self.g_ub, g_ub)


        # Read Symbolic Paramaters
        for i, param in enumerate(self.param_ca_dict):
            if i == 0:
                self._param_ca = self.param_ca_dict[param]["sym_param"]
            else:
                self._param_ca = ca.vertcat(self._param_ca, self.param_ca_dict[param]["sym_param"].reshape((-1,1)))

        # Solver settings
        options = {}
        options["ipopt.acceptable_tol"] = 1e-3
        if not verbose:
            options["ipopt.print_level"] = 0
            options["print_time"] = 0

        # Define solver
        self.solver = ca.nlpsol('solver', 'ipopt', {'x': self.x, 'f': self.objective.eval_cost(self.x), 'g': g, 'p': self._param_ca}, options)
    

        
    def solve(self):
        # Read Symbolic Paramaters
        for i, param in enumerate(self.param_ca_dict):
            if i == 0:
                self._param_num = self.param_ca_dict[param]["num_param"]
            else:
                self._param_num = ca.vertcat(self._param_num, self.param_ca_dict[param]["num_param"].reshape((-1,1), order='F'))

        result = self.solver(x0=self.x_init,
                             lbg=self.g_lb,
                             ubg=self.g_ub,
                             lbx=self.l_joint_limits.reshape((-1,1), order='F'),
                             ubx=self.u_joint_limits.reshape((-1,1), order='F'),
                             p=self._param_num)

        success_flag = self.solver.stats()["success"]
        success_msg = self.solver.stats()["return_status"]
        return  result['x'], success_flag
    

    def add_objective_function(self):
        # Define parameter sym variable
        name = "objective_param"
        self.param_ca_dict[name] =  {
            "sym_param" : ca.SX.sym(name, self.x_dim, 1),
            "num_param" : np.zeros((self.x_dim, 1))
            }
        # Create objective function
        self.objective = DistToHome(q_home=self.param_ca_dict[name]["sym_param"] ,
                                    n_dofs=self.x_dim)
      
    def add_position_constraint(self, tolerance=0.0):
        # Define parameter sym variable
        name = "position_g_param"
        self.param_ca_dict[name] =  {
            "sym_param" : ca.SX.sym(name, 4, 4),
            "num_param" : np.eye(4)
            }
        # Create constraint
        self.g_list.append(PositionConstraint(robot=self._robot_model,
                                              q_ca=self.x,
                                              paramca_T_W_Ref=self.param_ca_dict[name]["sym_param"],
                                              tolerance=tolerance))


    def add_orientation_constraint(self, tolerance=0.0):
        # Define parameter sym variable
        name = "orientation_g_param"
        self.param_ca_dict[name] =  {
            "sym_param" : ca.SX.sym(name, 4, 4),
            "num_param" : np.eye(4)
            }
        # Create constraint
        self.g_list.append(OrientationConstraint(robot=self._robot_model,
                                                 q_ca=self.x,
                                                 paramca_T_W_Ref=self.param_ca_dict[name]["sym_param"],
                                                 tolerance=tolerance))



    

    def add_collision_constraint(self, name, link_names, r_link=0.5, r_obst=0.2, tolerance=0.01):
        # Define parameter sym variable
        self.param_ca_dict[name] =  {
            "sym_param" : ca.SX.sym(name, 4, 4),
            "num_param" : np.eye(4)
            }
        # Create constraint
        for link in link_names:
            self.g_list.append(CollisionConstraint(robot=self._robot_model,
                                                    q_ca=self.x,
                                                    waypoint_ID=0,
                                                    param_obst_ca=self.param_ca_dict[name]["sym_param"],
                                                    link_name=link,
                                                    r_link= r_link,
                                                    r_obst= r_obst,
                                                    tolerance=tolerance))

