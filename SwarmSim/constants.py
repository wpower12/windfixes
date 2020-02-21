# Size of time step
DT = 0.005

# Drone 
DRONE_MASS = 1.0
MAX_VEL    = 1.0
MAX_ACC    = 1.0
MAX_JERK   = 1.0 # Max value that can be applied to acc in a dt

# Swarm
SEPARATION = 0.5 # Similarity matrix sparsification threshold

# Constants for the PID controller
PID_P = 30.0
PID_I = 0.05
PID_D = 125.0

# Parameters for Wind Distribution
START_P     = 0.02
LENGTH_MEAN = 20
LENGTH_VAR  = 5
ANGLE_MEAN  = 0.5236 # Should be in radians.
ANGLE_VAR   = 0.7854
ANGLE_ZENITH_MEAN = 0 # In radians
ANGLE_ZENITH_VAR = 0.5236
MAG_MEAN    = 5.0
MAG_VAR     = 2.5

MAX_SCALING = 1.0
MIN_SCALING = 0.8

# Expansion Procedure Parameters
TEST_VAR_RADIUS = 2.0
TARGET_EPSILON  = 0.1
EXP_OFF       = 0
EXP_HOVER     = 1
EXP_EXPANDING = 2

# Model Parameters
NUM_REGRESSORS = 1
WINDOW_SIZE = 5 # How large each 'temporal window' is