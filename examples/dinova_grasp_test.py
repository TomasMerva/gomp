from grasp_planning import GOMP
import numpy as np
import time
import os

# Robot urdf
absolute_path = os.path.dirname(os.path.abspath(__file__))
URDF_FILE = absolute_path + "/assets/dingo_kinova_gripper.urdf"
# Object pose
T_W_Obj = np.array([[ 1,     0.0,   0.0,  2.35231148],
                    [ 0.0,   1,     0,    -1.78948217],
                    [ 0.,    0.0,   1,    0.71829593],
                    [ 0.,    0.,    0.,   1.        ]], dtype=float)
# Current robot's state
q_current = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.57, 1.57,  1.57, 1.57], dtype=float)
# Obstacle's pose
T_W_Obst = np.eye(4)
T_W_Obst[:3,3] = np.array([-1., 0.4, 0.15]).T


"""
Planner
"""
n_waypoints = 3
grasp_waypoint_ID = 2
planner = GOMP(n_waypoints=n_waypoints, 
               urdf=URDF_FILE, 
               roll_obj_grasp=np.pi/4,
               root_link='world', 
               end_link='arm_tool_frame')


planner.set_init_guess(q_current)
planner.set_boundary_conditions(q_start=q_current)
planner.add_acc_objective_function()

# Create constraints for grasping
planner.add_grasp_pos_constraint(name="g_grasp_pos_1",
                                 waypoint_ID=grasp_waypoint_ID,
                                 tolerance=0.0)
planner.add_grasp_rot_dof_constraint(name="g_grasp_rot_1",
                                    waypoint_ID=grasp_waypoint_ID,
                                    theta = 0.0,
                                    axis="x")
# Create constraints for collision avoidance
active_links = ['chassis_link', 
                'arm_tool_frame']
g_collision_names = []
for i in range(n_waypoints):
    # planner.add_z_pos_constraint(name="g_zheight_" + str(i),
    #                              waypoint_ID=i,
    #                              z_height=0.1,
    #                              tolerance=0.01)
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
planner.setup_problem(verbose=True)

"""
Main loop
"""
start = time.time()
# 1. update constraint parameters (new object and obstacle pose)
constraint_param_dict = {
    "g_grasp_pos_1" : T_W_Obj,
    "g_grasp_rot_1" : T_W_Obj
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
for t in range(n_waypoints):
    T = planner._robot_model.eval_fk(x[:,t])
    print(f"Waypoint {t}\n{T}")
#     print(f"[[ {T[0,0]}, {T[0,1]}, {T[0,2]}, {T[0,3]}],\n \
# [{T[1,0]}, {T[1,1]}, {T[1,2]}, {T[1,3]}],\n \
# [{T[2,0]}, {T[2,1]}, {T[2,2]}, {T[2,3]}],\n \
# [{T[3,0]}, {T[3,1]}, {T[3,2]}, {T[3,3]}]]")


