from importlib.metadata import version
__version__ = version(__name__)

from grasp_planning.solver.gomp_planner import GOMP
from grasp_planning.solver.ik_optim import IK_OPTIM