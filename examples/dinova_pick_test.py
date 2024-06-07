from grasp_planning import PICK
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
n_waypoints = 4 # needs to be more than 3 for now
waypoint_dict = {
    # "current_ID" : 0,
    "pregrasp_ID" : 1,
    "grasp_ID" : 2,
    "postgrasp_ID" : 3,
}
theta = np.pi/2

planner = PICK(n_waypoints=n_waypoints, 
               urdf=URDF_FILE, 
               roll_obj_grasp=np.pi/2,
               root_link='world', 
               end_link='arm_tool_frame')
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
planner.add_grasp_rot_dof_constraint(name="g_grasp_rot_1",
                                    waypoint_ID=waypoint_dict["grasp_ID"],
                                    theta = theta,
                                    axis="x")
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
planner.setup_problem(verbose=True)



"""
Main loop
"""
start = time.time()
# 1. update constraint parameters (new object, obstacle pose, ...)
constraint_param_dict = {
    "g_grasp_pos_1" : T_W_Obj,
    "g_grasp_rot_1" : T_W_Obj
}
planner.update_constraints_params(constraint_param_dict)
# 2. Call solver
x, solver_flag = planner.solve() #decision variables = q + manip_frame (6,1)
end = time.time()


# Logging
print("============")
print(f"Computational time: {end-start}" )
print(f"Solver status: {solver_flag}" )
for t in range(n_waypoints):
    T = planner._robot_model.eval_fk(x[:planner.n_dofs,t])
    print(f"Waypoint {t}\n{T}")







