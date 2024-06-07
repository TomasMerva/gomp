from grasp_planning import PICK
import numpy as np
import time
import os

# Mug's pose
# T_W_Obj = np.array([[-0.71929728, -0.69467357,  0.0063291,  -2.35231148],
#                     [ 0.69430406, -0.71916348, -0.02730871,  1.78948217],
#                     [ 0.0235223,  -0.01524876,  0.99960701,  0.71829593],
#                     [ 0.,         0.,           0.,          1.        ]], dtype=float)
"""
Metadata
"""
absolute_path = os.path.dirname(os.path.abspath(__file__))
URDF_FILE = absolute_path + "/assets/dingo_kinova_gripper.urdf"
T_W_Obj = np.array([[ 1,     0.0,   0.0,  2.35231148],
                    [ 0.0,   1,     0,    -1.78948217],
                    [ 0.,    0.0,   1,    0.71829593],
                    [ 0.,    0.,    0.,   1.        ]], dtype=float)
# Current robot's state
q_current = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.57, 1.57,  1.57, 1.57], dtype=float)



"""
Planner
"""
# w1 = pregrasp
# w2 = grasp
# w3 = postgrasp
n_waypoints = 10 # needs to be more than 3 for now
waypoint_dict = {
    # "current_ID" : 0,
    "pregrasp_ID" : 7,
    "grasp_ID" : 8,
    "postgrasp_ID" : 9,
}
planner = PICK(n_waypoints=n_waypoints, 
               urdf=URDF_FILE, 
               theta=np.pi/2 , 
               roll_obj_grasp=np.pi/2,
               root_link='world', 
               end_link='arm_tool_frame')
print("Planner\n------")
planner.set_init_guess(q_current)
planner.set_boundary_conditions(q_start=planner.x_init[:15, 0])
planner.add_acc_objective_function()

"""
Constraints
"""
# Grasp waypoint
planner.add_grasp_pos_constraint(name="g_grasp_pos_1",
                                 waypoint_ID=waypoint_dict["grasp_ID"],
                                 tolerance=0.0)
planner.add_grasp_rot_constraint(name="g_grasp_rot_1",
                                 waypoint_ID=waypoint_dict["grasp_ID"],
                                 tolerance=0)
# Pregrasp waypoint
planner.add_pregrasp_constraint(name="g_pregrasp",
                                waypoint_ID=waypoint_dict["pregrasp_ID"],
                                grasp_waypoint_ID=waypoint_dict["grasp_ID"],
                                offset=-0.2,
                                tolerance=0.0)
# Postgrasp waypoint
planner.add_pregrasp_constraint(name="g_postgrasp",
                                waypoint_ID=waypoint_dict["postgrasp_ID"],
                                grasp_waypoint_ID=waypoint_dict["grasp_ID"],
                                offset=-0.1,
                                tolerance=0.0)

for waypoint, idx in waypoint_dict.items():
    planner.add_manipulation_constraint(name=waypoint,
                                        waypoint_ID=idx)

"""
Constraints parameter
"""
constraint_param_dict = {
    "g_grasp_pos_1" : T_W_Obj,
    "g_grasp_rot_1" : T_W_Obj
}
# for g_col in g_collision_names:
#     constraint_param_dict[g_col] = T_W_Obst[:3,3]
planner.update_constraints_params(constraint_param_dict)


planner.setup_problem(verbose=True)

for i in range(2):
    planner.update_constraints_params(constraint_param_dict)
    x, solver_flag = planner.solve()
    planner.x_init = x.reshape((-1,1))

    # T_W_Pre = planner._robot_model.eval_fk(x[:9, waypoint_dict["pregrasp_ID"]])
    # T_W_Grasp = planner._robot_model.eval_fk(x[:9, waypoint_dict["grasp_ID"]])
    # T_W_Post = planner._robot_model.eval_fk(x[:9, waypoint_dict["postgrasp_ID"]])

    # print("T_W_Pre\n", T_W_Pre)
    # print("T_W_Grasp\n", T_W_Grasp)
    # print("T_W_Post\n", T_W_Post)
    # print(f"Results\n")
    for t in range(n_waypoints):
        T = planner._robot_model.eval_fk(x[:9,t])
        print(f"Waypoint {t}")
        print(f"[[ {T[0,0]}, {T[0,1]}, {T[0,2]}, {T[0,3]}],\n \
[{T[1,0]}, {T[1,1]}, {T[1,2]}, {T[1,3]}],\n \
[{T[2,0]}, {T[2,1]}, {T[2,2]}, {T[2,3]}],\n \
[{T[3,0]}, {T[3,1]}, {T[3,2]}, {T[3,3]}]]")







