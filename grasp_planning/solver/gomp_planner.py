import numpy as np
import casadi as ca
import spatial_casadi as sc
from scipy.spatial.transform import Rotation as R
from grasp_planning.solver.robot_model import RobotKinematicModel
from grasp_planning.cost.costs import *
from grasp_planning.constraints.constraints import *

class GOMP():
    def __init__(self, n_waypoints, urdf, roll_obj_grasp = np.pi, root_link='world', end_link='tool_frame') -> None:
        # Init variables
        self.T_W_Grasp = None
        self.T_W_Obj = None
        self.T_Obj_Grasp = np.eye(4, dtype=float)
        
        # Kinematics
        self._robot_model = RobotKinematicModel(urdf, root_link, end_link)
        self.n_dofs = self._robot_model.n_dofs
        self.manipulation_frame_dim = 6
        self.x_dim = self.n_dofs
        self.n_waypoints = n_waypoints
        self.x_casadi_dim = self.x_dim *self.n_waypoints
        self.theta = 0.0
        self.roll_obj_grasp = roll_obj_grasp
        
        # Optimization
        self.x = ca.SX.sym("x", self.x_dim, self.n_waypoints)
        self.x_init = None
        
        self._objective = None
        self.l_joint_limits, self.u_joint_limits = None, None
        self.g_list = []
        
        self.param_ca_dict = dict()
        self._param_ca = None
        self._param_num = None


    def update_grasp_DOF(self, theta: float, degrees=True) -> None:
        """
        Object has to have z-axis aligned with the world frame
        """
        if degrees:
            self.theta = np.deg2rad(theta)
        else:
            self.theta = theta
        # Create static grasp's frame with respect to object's frame
        self.T_Obj_Grasp = np.eye(4, dtype=float)
        self.T_Obj_Grasp[:3,:3] = R.from_euler('xyz', [0, self.roll_obj_grasp, 0], degrees=False).as_matrix()

        # Lower bound with respect to the world frame
        self.T_W_Grasp = self.T_W_Obj @ self.T_Obj_Grasp 


    def update_object_pose(self, T_W_Obj: np.array) -> None:
        """
        Update object and grasp poses
        """
        self.T_W_Obj = T_W_Obj

    def set_init_guess(self, q):
        self.x_init = np.tile(q, self.n_waypoints).reshape((-1,1))

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

        if q_end != None:
            print("Final boundary condition is not implemented.")
    
    

    def setup_problem(self, verbose=False):
        # assert self.T_Obj_Grasp != None, "Grasp DOF is not set. Set additional degree of freedom around object."
        # assert self.T_W_Obj != None, "Object pose is not set."
        # 1. create decision variable
        x_ca_flatten = self.x.reshape((-1,1))
        
        # assert self.x_init != None, "Initial guess is missing. Set up one using `set_init_guess(q)`."

        g, self.g_lb, self.g_ub = ca.vertcat(), ca.vertcat(), ca.vertcat()
        for g_term in self.g_list:
            gt, g_lb, g_ub = g_term.get_constraint()
            g = ca.vertcat(g, gt)
            self.g_lb = ca.vertcat(self.g_lb, g_lb)
            self.g_ub = ca.vertcat(self.g_ub, g_ub)


        # Read Symbolic Paramaters
        for param in self.param_ca_dict:
            if self.param_ca_dict[param]["sym_param"] is not None:
                if self._param_ca is None:
                    self._param_ca = self.param_ca_dict[param]["sym_param"].reshape((-1,1))
                else:
                    self._param_ca = ca.vertcat(self._param_ca, self.param_ca_dict[param]["sym_param"].reshape((-1,1)))
        options = {}
        options["ipopt.acceptable_tol"] = 1e-3
        if not verbose:
            options["ipopt.print_level"] = 0
            options["print_time"] = 0

        self.solver = ca.nlpsol('solver', 'ipopt', {'x': x_ca_flatten, 'f': self.objective.eval_cost(x_ca_flatten), 'g': g, 'p': self._param_ca}, options)
    

        
    def solve(self):
        # Replace symbolic variables with numeric values
        self._param_num = None
        for param in self.param_ca_dict:
            if self.param_ca_dict[param]["num_param"] is not None:
                if self._param_num is None:
                    self._param_num = self.param_ca_dict[param]["num_param"].reshape((-1,1), order='F')
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

        temp_x = result['x'].reshape((self.n_dofs, self.n_waypoints))

        return  result['x'].reshape((self.n_dofs, self.n_waypoints)), success_flag


    def add_acc_objective_function(self):
        # Define parameter sym variable
        name = "objective_param"
        self.param_ca_dict[name] =  {
            "sym_param" : None,
            "num_param" : None,
            "grasp" : False
            }
        # Create objective function
        self.objective = SquaredAccCost(self.n_waypoints, self.x_dim, manip_frame=False)
    
    def add_dist_home_objective_function(self):
        # Define parameter sym variable
        name = "objective_param"
        self.param_ca_dict[name] =  {
            "sym_param" : ca.SX.sym(name, self.x_casadi_dim, 1),
            "num_param" : np.zeros((self.x_casadi_dim, 1)),
            "grasp" : False
            }
       
        self.objective = DistToHome(q_home=self.param_ca_dict[name]["sym_param"],
                                    n_dofs=self.x_casadi_dim)
    

    def update_constraints_params(self, content_dict):
        for g_name, g_param in content_dict.items():
            if self.param_ca_dict[g_name]["grasp"]:
                self.T_Obj_Grasp = np.eye(4, dtype=float)
                self.T_Obj_Grasp[:3,:3] = R.from_euler('xyz', [0, self.roll_obj_grasp, 0], degrees=False).as_matrix()
                self.T_W_Obj = g_param
                self.T_W_Grasp = self.T_W_Obj @ self.T_Obj_Grasp
                self.param_ca_dict[g_name]["num_param"] = self.T_W_Grasp
            else:
                self.param_ca_dict[g_name]["num_param"] = g_param
        
        return self.T_W_Grasp

    def add_collision_constraint(self, name, waypoint_ID, child_link, r_link=0.5, r_obst=0.2, tolerance=0.01):
        # Define parameter sym variable
        self.param_ca_dict[name] =  {
            "waypoint_ID" : waypoint_ID,
            "sym_param" : ca.SX.sym(name, 3, 1),
            "num_param" : np.zeros((3,1)),
            "tolerance" : tolerance,
            "child_link" : child_link,
            "r_link" : r_link,
            "r_obst" : r_obst,
            "grasp" : False,
            }
        self.g_list.append(CollisionConstraint(robot=self._robot_model,
                                                q_ca=self.x,
                                                waypoint_ID=waypoint_ID,
                                                param_obst_ca=self.param_ca_dict[name]["sym_param"],
                                                link_name=child_link,
                                                r_link= r_link,
                                                r_obst= r_obst,
                                                tolerance=tolerance))

    def add_grasp_pos_constraint(self, name, waypoint_ID, tolerance=0.0):
        # Define parameter sym variable
        self.param_ca_dict[name] =  {
            "waypoint_ID" : waypoint_ID,
            "sym_param" : ca.SX.sym(name, 4, 4),
            "num_param" : np.eye(4),
            "tolerance" : tolerance,
            "grasp" : True
            }
        # Create
        self.g_list.append(GraspPosConstraint(robot=self._robot_model, 
                                              q_ca=self.x, 
                                              waypoint_ID=waypoint_ID, 
                                              paramca_T_W_Grasp=self.param_ca_dict[name]["sym_param"],
                                              tolerance=tolerance))

    def add_grasp_rot_constraint(self, name, waypoint_ID, theta, tolerance=0.1):
        # Define parameter sym variable
        self.param_ca_dict[name] =  {
            "waypoint_ID" : waypoint_ID,
            "sym_param" : ca.SX.sym(name, 4, 4),
            "num_param" : np.eye(4),
            "tolerance" : tolerance,
            "theta" : theta,
            "grasp" : True
            }
        self.theta = theta
        self.g_list.append(GraspRotConstraint(robot=self._robot_model, 
                                              q_ca=self.x, 
                                              waypoint_ID=waypoint_ID, 
                                              paramca_T_W_Grasp=self.param_ca_dict[name]["sym_param"], 
                                              theta=self.theta, 
                                              tolerance=tolerance))


    def add_grasp_rot_dof_constraint(self, name, waypoint_ID, theta, axis="x"):
        # Define parameter sym variable
        self.param_ca_dict[name] =  {
            "waypoint_ID" : waypoint_ID,
            "sym_param" : ca.SX.sym(name, 4, 4),
            "num_param" : np.eye(4),
            "theta" : theta,
            "axis" : axis,
            "grasp" : True
            }
        self.theta = theta
        self.g_list.append(AngleBetweenVectorsConstraint(
                                                robot=self._robot_model,
                                                q_ca=self.x,
                                                waypoint_ID=waypoint_ID, 
                                                paramca_T_W_Grasp=self.param_ca_dict[name]["sym_param"], 
                                                theta=self.theta,
                                                axis=axis))

    def add_z_pos_constraint(self, name, waypoint_ID, z_height, tolerance=0.01):
        self.param_ca_dict[name] =  {
            "waypoint_ID" : waypoint_ID,
            "sym_param" : None,
            "num_param" : None,
            "z_height" : z_height,
            "tolerance" : tolerance,
            "grasp" : False
            }
        self.g_list.append(ZheightConstraint(robot=self._robot_model,
                                             q_ca=self.x,
                                             waypoint_ID=waypoint_ID, 
                                             z_height = z_height,
                                             tolerance=tolerance))