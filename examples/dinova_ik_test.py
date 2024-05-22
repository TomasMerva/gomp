from grasp_planning import IK_OPTIM
import numpy as np
import time
import os

"""
Model
"""
# URDF model
absolute_path = os.path.dirname(os.path.abspath(__file__))
URDF_FILE = absolute_path + "/assets/dingo_kinova_gripper.urdf"

"""
Default poses and configurations
"""
# Desired pose
T_W_Ref = np.array([[-7.02597833e-01, -2.96854396e-04, -7.11587097e-01, -2.34231147e+00],
                    [-2.46720383e-02,  9.99408826e-01,  2.39434383e-02,  1.77948216e+00],
                    [ 7.11159318e-01,  3.43789119e-02, -7.02189800e-01,  7.08295920e-01],
                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]], dtype=float)
# Obstacle's pose
T_W_Obst = np.eye(4)
T_W_Obst[:3,3] = np.array([-1., 0.4, 0.15]).T
# Current robot's state
q_home = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.57, 1.57,  1.57, 1.57], dtype=float)


"""
Solver
"""
# Create IK solver
planner = IK_OPTIM(urdf=URDF_FILE,
                   root_link = 'world', 
                   end_link  = 'arm_tool_frame')
planner.set_init_guess(q_home)
planner.set_boundary_conditions() # joint limits

planner.add_objective_function(name="objective")
planner.add_position_constraint(name="g_position", tolerance=0)
planner.add_orientation_constraint(name="g_rotation", tolerance=0.01)


# Define collision constraint for each link
active_links = ['chassis_link', 
                'arm_base_link', 
                'arm_shoulder_link', 
                'arm_arm_link', 
                'arm_forearm_link', 
                'arm_lower_wrist_link',
                'arm_upper_wrist_link', 
                'arm_end_effector_link', 
                'arm_tool_frame']
planner.add_collision_constraint(name="sphere_col",
                                 link_names=active_links, 
                                 r_link=0.2,
                                 r_obst=0.2,
                                 tolerance=0.01)
planner.param_ca_dict["sphere_col"]["num_param"] = T_W_Obst[:3,3]


# Formulate problem
planner.setup_problem(verbose=False)



planner.param_ca_dict["objective"]["num_param"] = q_home  # home configuration
planner.param_ca_dict["g_position"]["num_param"] = T_W_Ref
planner.param_ca_dict["g_rotation"]["num_param"] = T_W_Ref



# Call IK solver
for i in range(10):
    # Compute new desired pose
    T_W_Ref = np.array([[-7.02597833e-01, -2.96854396e-04, -7.11587097e-01, -2.34231147e+00-i*0.23],
                    [-2.46720383e-02,  9.99408826e-01,  2.39434383e-02,  1.77948216e+00-i*0.23],
                    [ 7.11159318e-01,  3.43789119e-02, -7.02189800e-01,  7.08295920e-01],
                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]], dtype=float)

    start = time.time()
    # Modify parameters during runtime
    planner.set_init_guess(q_home)
    planner.param_ca_dict["objective"]["num_param"] = q_home # setting home configuration
    planner.param_ca_dict["g_position"]["num_param"] = T_W_Ref
    planner.param_ca_dict["g_rotation"]["num_param"] = T_W_Ref
    # Call solver
    x, solver_flag = planner.solve()
    end = time.time()
    # Logging
    print(f"Iteration {i}")
    print(f"Computational time: {end-start}" )
    print(f"Solver status: {solver_flag}" )

# T_W_Grasp = planner._robot_model.eval_fk(x)
# print(T_W_Grasp)