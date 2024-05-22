from grasp_planning import GOMP
import numpy as np
import time
import os

# Mug's pose
T_W_Obj = np.array([[-0.71929728, -0.69467357,  0.0063291,  -2.35231148],
                    [ 0.69430406, -0.71916348, -0.02730871,  1.78948217],
                    [ 0.0235223,  -0.01524876,  0.99960701,  0.71829593],
                    [ 0.,         0.,           0.,          1.        ]], dtype=float)

T_W_Place = np.array([[-0.71929728, -0.69467357,  0.0063291,   5.35231148],
                    [ 0.69430406, -0.71916348, -0.02730871,  5.78948217],
                    [ 0.0235223,  -0.01524876,  0.99960701,  0.71829593],
                    [ 0.,         0.,           0.,          1.        ]], dtype=float)


# Obstacle's pose
T_W_Obst = np.eye(4)
T_W_Obst[:3,3] = np.array([-1., 0.4, 0.15]).T

# Current robot's state
q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.57, 1.57,  1.57, 1.57], dtype=float)
q_current= np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0, 0,  0, 0], dtype=float)

absolute_path = os.path.dirname(os.path.abspath(__file__))
URDF_FILE = absolute_path + "/assets/dingo_kinova_gripper.urdf"


n_waypoints = 10 # needs to be more than 3 for now
grasp_waypoint_ID = 3
place_waypoint_ID = 9
planner = GOMP(n_waypoints=n_waypoints, 
               urdf=URDF_FILE, 
               theta=np.pi/4 , 
               roll_obj_grasp=np.pi/2,
               root_link='world', 
               end_link='arm_tool_frame')




planner.set_init_guess(q_init)
planner.set_boundary_conditions(q_start=q_current)
planner.add_objective_function()
# Create constraints for grasping
planner.add_grasp_pos_constraint(name="g_grasp_pos_1",
                                 waypoint_ID=grasp_waypoint_ID,
                                 tolerance=0.0)
planner.add_grasp_rot_constraint(name="g_grasp_rot_1",
                                 waypoint_ID=grasp_waypoint_ID,
                                 tolerance=0.02)
# Create constraints for placing
planner.add_grasp_pos_constraint(name="g_place_pos_1",
                                 waypoint_ID=place_waypoint_ID,
                                 tolerance=0.0)
planner.add_grasp_rot_constraint(name="g_place_rot_1",
                                 waypoint_ID=place_waypoint_ID,
                                 tolerance=0.02)
# Create constraints for collision avoidance
active_links = ['chassis_link', 
                'arm_tool_frame']
g_collision_names = []
for i in range(n_waypoints):
    planner.add_z_pos_constraint(name="g_zheight_" + str(i),
                                 waypoint_ID=i,
                                 z_height=0.1,
                                 tolerance=0.01)
    for link in active_links:
        name = "g_col_"+link + "_" + str(i)
        g_collision_names.append(name)
        planner.add_collision_constraint(name=name,
                                         waypoint_ID=i, 
                                         child_link=link, 
                                         r_link=0.5,
                                         r_obst=0.2,
                                         tolerance=0.01)
        
# Setup problem
planner.setup_problem(verbose=False)

"""
Main loop
"""
start = time.time()
# 1. update constraint parameters
constraint_param_dict = {
    "g_grasp_pos_1" : T_W_Obj,
    "g_grasp_rot_1" : T_W_Obj,
    "g_place_pos_1" : T_W_Place,
    "g_place_rot_1" : T_W_Place
}
for g_col in g_collision_names:
    constraint_param_dict[g_col] = T_W_Obst[:3,3]
planner.update_constraints_params(constraint_param_dict)

# 2. Call solver
x, solver_flag = planner.solve()

end = time.time()

# Logging
print("============")
print(f"Computational time: {end-start}" )
print(f"Solver status: {solver_flag}" )
# print(f"Results\n")
# for t in range(n_waypoints):
#     T = planner._robot_model.eval_fk(x[:,t])
#     print(f"Waypoint {t}")
#     print(f"[[ {T[0,0]}, {T[0,1]}, {T[0,2]}, {T[0,3]}],\n \
# [{T[1,0]}, {T[1,1]}, {T[1,2]}, {T[1,3]}],\n \
# [{T[2,0]}, {T[2,1]}, {T[2,2]}, {T[2,3]}],\n \
# [{T[3,0]}, {T[3,1]}, {T[3,2]}, {T[3,3]}]]")


