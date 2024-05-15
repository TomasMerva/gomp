This package provides grasp and motion planning using CasADi and IPOPT. 


## Usage
```python
from grasp_planning import GOMP
import numpy as np
import time
import os

# Mug's pose
T_W_Obj = np.array([[-0.71929728, -0.69467357,  0.0063291,  -2.35231148],
                    [ 0.69430406, -0.71916348, -0.02730871,  1.78948217],
                    [ 0.0235223,  -0.01524876,  0.99960701,  0.71829593],
                    [ 0.,         0.,           0.,          1.        ]], dtype=float)
# Obstacle's pose
T_W_Obst = np.eye(4)
T_W_Obst[:3,3] = np.array([1.86, 0.4, 0.15]).T

# Current robot's state
q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.57, 1.57,  1.57, 1.57], dtype=float)

absolute_path = os.path.dirname(os.path.abspath(__file__))
URDF_FILE = absolute_path + "/assets/dingo_kinova_gripper.urdf"


num_waypoints = 3 # needs to be more than 3 for now
theta = np.pi/2 #Degree of freedom around grasp pose
planner = GOMP(num_waypoints, URDF_FILE, theta, 'world', 'arm_tool_frame')
planner.set_init_guess(q_init)
planner.set_boundary_conditions(q_start=q_init)
planner.add_grasp_constraint(waypoint_ID=2, tolerance=0.01)
for i in range(num_waypoints):
    planner.add_collision_constraint(waypoint_ID=i, 
                                    child_link="chassis_link", 
                                    r_link=0.5,
                                    r_obst=0.2,
                                    tolerance=0.01)
planner.setup_problem(verbose=False)


start = time.time()
planner.update_constraints_params(T_W_Obj, T_W_Obst)
x, solver_flag = planner.solve()
end = time.time()
print(f"Computational time: {end-start}" )
print(f"Solver status: {solver_flag}" )
```

