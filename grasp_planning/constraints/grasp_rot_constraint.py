import numpy as np
import casadi as ca
import spatial_casadi as sc
from grasp_planning.constraints.constraint_template import Constraint

# class GraspRotConstraint(Constraint):
#     def __init__(self, robot, q_ca, waypoint_ID, param_grasp_ca, theta=0.0, tolerance=0.1) -> None:
#         super().__init__() 
#         # q = q_ca[:n_dof, waypoint_ID] -> joint space of the robot (decision variable)
#         # manip_frame = q_ca[n_dof:, waypoint_ID] -> manipulation frame (decision variable)
#         # manip frame_pos - T_W_Grasp_pos = 0
#         self._robot = robot
#         self.theta = theta
#         self.tolerance = tolerance
#         self.param_grasp_ca = param_grasp_ca
        
#         R_W_EEF = self._robot.compute_fk_ca(q_ca[:,waypoint_ID])
#         R_G_EEF = self.param_grasp_ca[:3,:3].T @ R_W_EEF[:3,:3]
#         Rpy_G_EEF = sc.Rotation.from_matrix(R_G_EEF).as_euler("xyz")

#         self.g = ca.vertcat(Rpy_G_EEF[0], Rpy_G_EEF[1], Rpy_G_EEF[2])
#         self.g_lb = ca.vertcat(-self.tolerance, -self.theta, -self.tolerance)
#         self.g_ub = ca.vertcat(self.tolerance, self.theta, self.tolerance)

#         self.g_eval = ca.Function('g_grasp_rot', [q_ca, param_grasp_ca], [self.g])

#     def get_constraint(self):
#         return self.g, self.g_lb, self.g_ub

#     def do_eval(self, q, T_W_Grasp):
#         return self.g_eval(q, T_W_Grasp)

class GraspRotConstraint(Constraint):
    def __init__(self, robot, q_ca, waypoint_ID, paramca_T_W_Grasp, theta=0.0, tolerance=0.1) -> None:
        super().__init__() 
        # q = q_ca[:n_dof, waypoint_ID] -> joint space of the robot (decision variable)
        # manip_frame = q_ca[n_dof:, waypoint_ID] -> manipulation frame (decision variable)
        # manip frame_pos - T_W_Grasp_pos = 0
        self._robot = robot
        self.theta = theta
        self.tolerance = tolerance
        
        R_W_EEF = self._robot.compute_fk_ca(q_ca[:,waypoint_ID])
        R_W_G = paramca_T_W_Grasp[:3,:3]
        R_G_EEF = (R_W_G.T) @ R_W_EEF[:3,:3]
        # Rpy_G_EEF = sc.Rotation.from_matrix(R_G_EEF).as_euler("xyz")

        # self.g = ca.vertcat(Rpy_G_EEF[0], Rpy_G_EEF[1], Rpy_G_EEF[2])
        # self.g_lb = ca.vertcat(-self.theta/2, -self.tolerance,  -self.tolerance)
        # self.g_ub = ca.vertcat(self.theta/2, self.tolerance,   self.tolerance)

        x_G = paramca_T_W_Grasp[:3,0] / ca.norm_2(paramca_T_W_Grasp[:3,0])
        # y_G = paramca_T_W_Grasp[:3,1] / ca.norm_2(paramca_T_W_Grasp[:3,1])
        # z_G = paramca_T_W_Grasp[:3,2] / ca.norm_2(paramca_T_W_Grasp[:3,2])
       

        x_EEF = R_W_EEF[:3,0] / ca.norm_2(R_W_EEF[:3,0])

        # self.g = ca.norm_2(x_G-x_EEF)
        self.g = ca.cross(x_G, x_EEF)
        # print(self.g.shape)
        # y_EEF = R_W_EEF[:3,1] / ca.norm_2(R_W_EEF[:3,1])
        # z_EEF = R_W_EEF[:3,2] / ca.norm_2(R_W_EEF[:3,2])

        # g_x = x_G.T @ R_G_EEF @ x_EEF
        # g_y = y_G.T @ R_G_EEF @ y_EEF
        # g_z = z_G.T @ R_G_EEF @ z_EEF

        # g_x = ca.dot(x_G.T, ca.dot(R_G_EEF, x_EEF))
        # g_y = ca.dot(y_G.T, ca.dot(R_G_EEF, y_EEF))
        # g_z = ca.dot(z_G.T, ca.dot(R_G_EEF, z_EEF))
        # self.g = g_x
        self.g_lb = ca.vertcat(-self.tolerance, -self.tolerance, -self.tolerance)
        self.g_ub = ca.vertcat(self.tolerance, self.tolerance, self.tolerance)


        # self.g =  z_EEF.T @ R_G_EEF @z_G
        # self.g_lb = np.cos(self.theta)
        # self.g_ub = np.cos(self.theta)

        # self.g = ca.dot(paramca_T_W_Grasp[:3,3], R_W_EEF[:3,3])
        # self.g_lb = -self.tolerance
        # self.g_ub = self.tolerance
        # self.g_lb = ca.vertcat(-self.theta, -self.tolerance, -self.tolerance)
        # self.g_ub = ca.vertcat(self.theta, self.tolerance, self.tolerance)
        # R_W_EEF = self._robot.compute_fk_ca(q_ca[:,waypoint_ID])
        # R_G_EEF = paramca_T_W_Grasp[:3,:3].T @ R_W_EEF[:3,:3]
        # # self.g = ca.trace(R_G_EEF[:2,:2])
        # self.g = R_G_EEF[2,2]
        # self.g_lb = 2*np.cos(self.tolerance) + 1.0
        # self.g_ub = ca.inf




        # self.g = ca.vertcat(Rpy_G_EEF[1], Rpy_G_EEF[2])

        # self.g_lb = ca.vertcat(-self.tolerance, -self.tolerance)
        # self.g_ub = ca.vertcat(self.tolerance, self.tolerance)


        self.g_eval = ca.Function('g_grasp_rot', [q_ca, paramca_T_W_Grasp], [self.g])

    def get_constraint(self):
        return self.g, self.g_lb, self.g_ub

    def do_eval(self, q, T_W_Grasp):
        return self.g_eval(q, T_W_Grasp), self.g_lb, self.g_ub
