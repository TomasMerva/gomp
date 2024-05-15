import numpy as np
import casadi as ca
import spatial_casadi as sc
from scipy.spatial.transform import Rotation as R
from grasp_planning.solver.robot_model import RobotKinematicModel
from grasp_planning.cost.costs import SquaredAccCost
from grasp_planning.constraints.constraints import *

class GOMP():
    def __init__(self, n_waypoints, urdf, theta=np.pi/4, root_link='world', end_link='tool_frame') -> None:
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
        self.theta = theta
        
        # Optimization
        self.x = ca.SX.sym("x", self.x_dim, self.n_waypoints)
        self.x_init = None
        
        self._objective = None
        self.l_joint_limits, self.u_joint_limits = None, None
        self.g_list = []

        self.param_grasp_ca = ca.SX.sym("param_grasp", 4, 4)
        self.param_obst_ca = ca.SX.sym("param_obst", 4, 4)
        self._collision_flag = False
        self.param_optim_ca = ca.vertcat()
        self.param_ca_list = []


    def update_grasp_DOF(self, theta: float, degrees=True) -> None:
        """
        Object has to have z-axis aligned with the world frame
        """
        self.T_Obj_Grasp = np.eye(4, dtype=float)
        if degrees:
            self.theta = np.deg2rad(theta)
            self.R_Obj_Grasp = R.from_euler('xyz', [180, 0, self.theta], degrees=True).as_matrix()
        else:
            self.theta = theta
            self.R_Obj_Grasp = R.from_euler('xyz', [np.pi, 0, self.theta], degrees=False).as_matrix()
        self.T_Obj_Grasp[:3,:3] = self.R_Obj_Grasp
        self.T_W_Grasp = self.T_W_Obj @ self.T_Obj_Grasp
        
        # print("T_W_Grasp\n", self.T_W_Grasp)


    def update_object_pose(self, T_W_Obj: np.array) -> None:
        """
        Update object and grasp poses
        """
        self.T_W_Obj = T_W_Obj

    def _grasp2rpy(self):
        """
        Convert T_W_Grasp into [x,y,z,r,p,y]
        """
        R_W_Grasp_rpy = sc.Rotation.from_matrix(self.T_W_Grasp[:3,:3]).as_euler("xyz") #convert rotation part of FK into rpy 
        self.T_W_Grasp_rpy = np.vstack((self.T_W_Grasp[:3,3][:,None], R_W_Grasp_rpy))
        return self.T_W_Grasp_rpy


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

        if q_end != None:
            print("Final boundary condition is not implemented.")
    
    

    def setup_problem(self, verbose=False):
        # assert self.T_Obj_Grasp != None, "Grasp DOF is not set. Set additional degree of freedom around object."
        # assert self.T_W_Obj != None, "Object pose is not set."
        # 1. create decision variable
        x_ca_flatten = self.x.reshape((-1,1))
        
        # assert self.x_init != None, "Initial guess is missing. Set up one using `set_init_guess(q)`."

        # Define objective function
        cost = SquaredAccCost(self.n_waypoints, self.x_dim, manip_frame=True)
        objective = cost.eval_cost(x_ca_flatten)

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

        self.solver = ca.nlpsol('solver', 'ipopt', {'x': x_ca_flatten, 'f': objective, 'g': g, 'p': self.param_optim_ca}, options)
    

        
    def solve(self):
        result = self.solver(x0=self.x_init,
                             lbg=self.g_lb,
                             ubg=self.g_ub,
                             lbx=self.l_joint_limits.reshape((-1,1), order='F'),
                             ubx=self.u_joint_limits.reshape((-1,1), order='F'),
                             p=self.params_optim_num)

        success_flag = self.solver.stats()["success"]
        success_msg = self.solver.stats()["return_status"]
        return  result['x'].reshape((self.n_dofs, self.n_waypoints)), success_flag

    def _add_grasp_pos_constraint(self, waypoint_ID, tolerance=0.0):
        self.g_list.append(GraspPosConstraint(self._robot_model, 
                                              self.x, 
                                              waypoint_ID, 
                                              self.param_grasp_ca, 
                                              tolerance))

    def _add_grasp_rot_constraint(self, waypoint_ID, tolerance=0.1):
        self.g_list.append(GraspRotConstraint(self._robot_model, 
                                              self.x, 
                                              waypoint_ID, 
                                              self.param_grasp_ca, 
                                              self.theta, 
                                              tolerance))

    def add_grasp_constraint(self, waypoint_ID, tolerance=0.0):
        self._add_grasp_pos_constraint(waypoint_ID, tolerance)
        self._add_grasp_rot_constraint(waypoint_ID, tolerance)
        self.param_ca_list.append(self.param_grasp_ca)

    def add_collision_constraint(self, waypoint_ID, child_link, r_link=0.5, r_obst=0.2, tolerance=0.01):
        self._collision_flag = True
        self.g_list.append(CollisionConstraint(robot=self._robot_model,
                                                q_ca=self.x,
                                                waypoint_ID=waypoint_ID,
                                                param_obst_ca=self.param_obst_ca,
                                                link_name=child_link,
                                                r_link= r_link,
                                                r_obst= r_obst,
                                                tolerance=tolerance))

    def update_constraints_params(self, T_W_Obj, T_W_Obst=None):
        self.update_object_pose(T_W_Obj)
        self.update_grasp_DOF(theta=self.theta, degrees=False)

        if self._collision_flag:
            self.params_optim_num = ca.vertcat(self.T_W_Grasp, T_W_Obst)
        else:
            self.params_optim_num = self.T_W_Grasp

