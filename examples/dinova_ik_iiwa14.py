from grasp_planning import IK_OPTIM
import numpy as np
import time
import os

# Desired pose
T_W_Ref = np.array([[-7.02597833e-01, -2.96854396e-04, -7.11587097e-01, -2.34231147e+00],
                    [-2.46720383e-02,  9.99408826e-01,  2.39434383e-02,  1.77948216e+00],
                    [ 7.11159318e-01,  3.43789119e-02, -7.02189800e-01,  7.08295920e-01],
                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]], dtype=float)
# Obstacle's pose
T_W_Obst = np.eye(4)
T_W_Obst[:3,3] = np.array([-1., 0.4, 0.15]).T

# Current robot's state
q_home = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1], dtype=float)

# URDF model
absolute_path = os.path.dirname(os.path.abspath(__file__))
URDF_FILE = absolute_path + "/assets/iiwa14.urdf"


# Create IK solver
planner = IK_OPTIM(urdf=URDF_FILE,
                   root_link = 'world', 
                   end_link  = 'iiwa_link_ee')
planner.set_home_config(q_home)
planner.set_init_guess(q_home)
planner.set_boundary_conditions() # joint limits
planner.add_position_constraint(tolerance=0.01)
planner.add_orientation_constraint(tolerance=0.01)
# Define collision constraint for each link
active_links = [f'iiwa_link_{i}' for i in range(8)]
active_links.append('iiwa_link_ee')
# for link in active_links:
#     planner.add_collision_constraint(child_link=link,
#                                      r_link=0.2,
#                                      r_obst=0.2,
#                                      tolerance=0.01)
# Formulate problem
planner.setup_problem(verbose=False)


# Call IK solver
for i in range(10):
    T_W_Ref = np.array([[-7.02597833e-01, -2.96854396e-04, -7.11587097e-01, -2.34231147e+00-i*0.23],
                    [-2.46720383e-02,  9.99408826e-01,  2.39434383e-02,  1.77948216e+00-i*0.23],
                    [ 7.11159318e-01,  3.43789119e-02, -7.02189800e-01,  7.08295920e-01],
                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]], dtype=float)

    start = time.time()
    planner.update_constraints_params(T_W_Ref=T_W_Ref,
                                    T_W_Obst=T_W_Obst)
    x, solver_flag = planner.solve()
    end = time.time()
    print(f"Iteration {i}")
    print(f"Computational time: {end-start}" )
    print(f"Solver status: {solver_flag}" )

# T_W_Grasp = planner._robot_model.eval_fk(x)
# print(T_W_Grasp)