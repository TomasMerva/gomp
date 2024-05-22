from grasp_planning import IK_OPTIM
import numpy as np
import time
import os
from scipy.spatial.transform import Rotation as R

# Desired pose (checked to be reachable for iiwa14):
# xee_NN: [ 0.52677347  0.35190244  0.48685025  0.81098249  0.17874092  0.44319898
# -0.33754081]
goal_position = np.array([[0.52677347,  0.35190244,  0.48685025]])
goal_orientation = np.array([0.81098249,  0.17874092,  0.44319898, -0.33754081])
r = R.from_quat(goal_orientation)
orientation_matrix = r.as_matrix()
T_W_Ref = np.eye(4)
T_W_Ref[:3,:3] = orientation_matrix
T_W_Ref[:3, 3] = goal_position
# T_W_Ref = np.concatenate([orientation_matrix, goal_position], axis=1)

# Obstacle's pose
T_W_Obst = np.eye(4)
T_W_Obst[:3,3] = np.array([-1., 0.4, 0.15]).T

# Current robot's state
q_home = np.array([0.0, 0.0, 1.5, 1.5, 0.0, 0.0, 0.0], dtype=float)

# URDF model
absolute_path = os.path.dirname(os.path.abspath(__file__))
URDF_FILE = absolute_path + "/assets/iiwa14.urdf"



# Create IK solver
planner = IK_OPTIM(urdf=URDF_FILE,
                   root_link = 'world', 
                   end_link  = 'iiwa_link_ee')
planner.set_init_guess(q_home)
planner.set_boundary_conditions() # joint limits

planner.add_objective_function(name="objective")
planner.add_position_constraint(name="g_position", tolerance=0)
planner.add_orientation_constraint(name="g_rotation", tolerance=0.01)


# Define collision constraint for each link
active_links = [f'iiwa_link_{i}' for i in range(8)]
active_links.append('iiwa_link_ee')
planner.add_collision_constraint(name="sphere_col",
                                 link_names=active_links, 
                                 r_link=0.2,
                                 r_obst=0.2,
                                 tolerance=0.01)
planner.param_ca_dict["sphere_col"]["num_param"] = T_W_Obst[:3,3]


# Formulate problem
planner.setup_problem(verbose=False)


# Call IK solver
for i in range(10):
    T_W_Ref[1, 3] = T_W_Ref[1, 3] - i * 0.03

    start = time.time()
    planner.set_init_guess(q_home)
    planner.param_ca_dict["objective"]["num_param"] = q_home # setting home configuration
    planner.param_ca_dict["g_position"]["num_param"] = T_W_Ref
    planner.param_ca_dict["g_rotation"]["num_param"] = T_W_Ref

    x, solver_flag = planner.solve()
    end = time.time()
    print(f"Iteration {i}")
    print(f"Computational time: {end-start}" )
    print(f"Solver status: {solver_flag}" )

# # print("========")
# # print("Desired T\n", T_W_Ref)

# T_W_Grasp = planner._robot_model.eval_fk(x)
# print("IK solver\n", T_W_Grasp)

