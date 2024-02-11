"""
File to keep track of all variables.
"""

############################ TRAINING DATA #############################

DATAPATH = "/home/ucaptp0/oasis-rt-surrogate/data/preprocessed_data/rnn_lw/dynamical"
CHECKPOINT_PATH = "/home/ucaptp0/oasis-rt-surrogate/checkpoints/rnn_lw/dynamical"

INPUT_VARS = [
    "Rho",
    "Temperature",
    "Pressure",
]

AUX_VARS = ["alb_surf_lw"] # sTemperature is automatically included in preprocessing script

TARGET_VARS = [
    "fnet_dn_lw_h",
    "fnet_up_lw_h",
]

############################ MODEL TRAINING ###########################

# Activation Functions
activ_RNN1 = "relu"
activ_Dense = "relu"
activ_RNN2 = "relu"
activ_RNN3 = "relu"
activ_output = "sigmoid"

EPOCHS = 100 # 100 
PATIENCE = 25
BATCH_SIZE = 4096 #128

NUM_NEURONS = 16
BASE_LR = 0.0001 
MAX_LR = 0.0005


