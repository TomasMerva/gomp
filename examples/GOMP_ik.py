import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from grasp_planning import IK_OPTIM

class IKGomp():
    def __init__(self, q_home=None):
        # Current robot's state
        if q_home is None:
            self.q_home = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1], dtype=float)
        else:
            self.q_home = q_home

    def construct_ik(self, urdf_path = "/../examples/urdfs/iiwa14.urdf"):
        """
        Construct IK solver
        """
        # URDF model
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        URDF_FILE = absolute_path + "/assets/iiwa14.urdf"
  
        # Create IK solver
        self.planner_gomp = IK_OPTIM(urdf=URDF_FILE,
                           root_link='iiwa_link_0',
                           end_link='iiwa_link_ee')
        q_home = np.array([0.0, 0.0, 1.5, 1.5, 0.0, 0.0, 0.0], dtype=float)

        self.q_current = np.array([2.96705976, -0.65804652,  1.02035138,  1.66700672,  0.79093408,  0.10747635, -0.91911763])

        self.planner_gomp.set_home_config(q_home)
        self.planner_gomp.set_init_guess(self.q_current)
        self.planner_gomp.set_boundary_conditions()  # joint limits
        self.planner_gomp.add_position_constraint(tolerance=0.0)
        self.planner_gomp.add_orientation_constraint(tolerance=0.0)
        # Define collision constraint for each link
        active_links = [f'iiwa_link_{i}' for i in range(8)]
        active_links.append('iiwa_link_ee')
        # for link in active_links:
        #     self.planner_gomp.add_collision_constraint(child_link=link,
        #                                      r_link=0.2,
        #                                      r_obst=0.2,
        #                                      tolerance=0.01)
        # Formulate problem
        self.planner_gomp.setup_problem(verbose=True)

    def construct_T_matrix(self, position:np.ndarray, orientation=None):
        """
        Transformation matrix from pose (position + quaternion)
        """
        try:
            r = R.from_quat(orientation)
            orientation_matrix = r.as_matrix()
        except:
            orientation_matrix = np.eye(3)
        position_transpose = position[:, np.newaxis]
        T = np.concatenate([orientation_matrix, position_transpose], axis=1)
        T= np.concatenate([T, np.array([[0, 0, 0, 1]])])
        return T

    def construct_T_matrices(self, positions_obsts:list):
        """
        Multiple transformation matrices
        """
        T_list = []
        for positions_obst in positions_obsts:
            T = self.construct_T_matrix(positions_obst)
            T_list.append(T)
        return T_list

    def call_ik(self, pose_d: np.ndarray, positions_obsts: list):
        """
        At run-time construct the IK solution
        """
        T_Ref = self.construct_T_matrix(pose_d[0:3], pose_d[3:7])
        T_Obsts = self.construct_T_matrices(positions_obsts)
        self.planner_gomp.update_constraints_params(T_W_Ref=T_Ref)
                                                    #T_W_Obst=T_Obsts[0])
        x, solver_flag = self.planner_gomp.solve()
        T_W_Grasp = self.planner_gomp._robot_model.eval_fk(x)
        print("IK solver\n", T_W_Grasp)
        return x.full().transpose()[0], solver_flag

if __name__ == '__main__':
    goal_pose = np.array([0.52677347,  0.35190244,  0.48685025, 0.81098249,  0.17874092,  0.44319898, -0.33754081])
    # goal_pose = np.array([0.52475053,  0.3556481,   0.4873063, 0.80763125 , 0.1876557, 0.4418802, -0.34243107])

    position_obst1 = np.array([-1., 0.4, 0.15])

    ik_gomp = IKGomp()


    ik_gomp.construct_ik()
    x, solver_flag  = ik_gomp.call_ik(goal_pose, positions_obsts=[position_obst1])

    print(f"Solution: {x}" )
    print(f"Solver status: {solver_flag}" )
    print(ik_gomp.q_current )
