"""
File to keep track of all variables.
"""


############################ TRAINING DATA #############################

DATAPATH = "/home/ucaptp0/oasis-rt-surrogate/data/preprocessed_data/rnn_sw/dynamical"
OPTICAL_VARS_PATH = '/home/ucaptp0/oasis-rt-surrogate/data/opacities_data'
CHECKPOINT_PATH = "/home/ucaptp0/oasis-rt-surrogate/checkpoints/rnn_sw/dynamical"

INPUT_VARS_DYNAMICAL = [
    "Rho",
    "Temperature",
    "Pressure",
]

AUX_VARS = ["cosz", "alb_surf_sw"]

TARGET_VARS = [
    "fnet_dn_sw_h",
    "fnet_up_sw_h",
]

############################ MODEL TRAINING ###########################

# Activation Functions
activ_RNN1 = "relu"
activ_Dense = "relu"
activ_RNN2 = "relu"
activ_RNN3 = "relu"
activ_output = "sigmoid"

EPOCHS = 100
PATIENCE = 25
BATCH_SIZE = 512 #512 #128

NUM_NEURONS = 16
BASE_LR = 0.0001 
MAX_LR = 0.0005


