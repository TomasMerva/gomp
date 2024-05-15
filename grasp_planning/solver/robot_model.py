import numpy as np
import casadi as ca
import spatial_casadi as sc
from scipy.spatial.transform import Rotation as R
from forwardkinematics import GenericURDFFk

class RobotKinematicModel():
    def __init__(self, urdf_file, root_link, end_link):
        with open(urdf_file, "r") as file:
            self.urdf = file.read()

        self.robot_fk = GenericURDFFk(
                            self.urdf,
                            root_link = root_link,
                            end_links= end_link
                        )
        self.root_link = root_link
        self.end_link = end_link
        self.n_dofs = self.robot_fk.n()
        self.joint_names = []
        self.joint_pos_limits = np.zeros((self.n_dofs, 2))
        self._read_joint_pos_limits()

    def compute_fk_ca(self, q_ca, end_link=None):
        if end_link == None:
            return self.robot_fk.casadi(q_ca, self.end_link, position_only=False)
        else:
            return self.robot_fk.casadi(q_ca, child_link=end_link, position_only=False)


    def compute_fk_rpy_ca(self, q_ca):
        fk_ca = self.compute_fk_ca(q_ca)
        R_ca_rpy = sc.Rotation.from_matrix(fk_ca[:3,:3]).as_euler("xyz") #convert rotation part of FK into rpy
        return ca.vertcat(fk_ca[:3,3], R_ca_rpy)

    def eval_fk(self, q):
        return self.robot_fk.numpy(q, self.end_link, position_only=False)

    def _read_joint_pos_limits(self):
        joints = self.robot_fk.robot.get_joint_info(self.root_link, self.end_link )
        i = 0
        for joint in joints:
            if joint.type in ["revolute", "prismatic"]:
                self.joint_names.append(joint.name)
                # self.joint_pos_limits.append([joint.limit.lower, joint.limit.upper])
                self.joint_pos_limits[i] = [joint.limit.lower, joint.limit.upper]
                i += 1


    def get_joint_pos_limits(self):
        return self.joint_pos_limits

    def get_link_names(self):
        return self.robot_fk.robot.link_names()
