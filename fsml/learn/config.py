from typing import List
import torch.nn as nn
import torch.optim as optim
import os.path as opath
import os 

# -----------------------------------------------------------------------------
# KFold Cross Validation Configurations
# -----------------------------------------------------------------------------
KF_SPLIT        : int  = 5
KF_POSSIBILITIES: List[int] = [3, 5, 7, 9]

# -----------------------------------------------------------------------------
# General Training and Testing Configurations
# -----------------------------------------------------------------------------
BATCH_SIZE        : int             = 20
NUM_EPOCHS        : int             = 250
ACCURACY_THRESHOLD: float           = 2.00
IMG_PATH          : str             = opath.join(os.getcwd(), "img")
MODEL_PATH        : str             = opath.join(os.getcwd(), "models")
OPTIMIZER         : optim.Optimizer = optim.Adam
CRITERION         : nn.Module       = nn.MSELoss
DATA_PATH         : str             = os.path.join(os.getcwd(), "data/meanstd/")


# -----------------------------------------------------------------------------
# General MLP predictor configuration and possibilities
# -----------------------------------------------------------------------------
MLP_NUM_HIDDEN_INPUT  : int = 5
MLP_NUM_HIDDEN_OUTPUT : int = 3
MLP_HIDDEN_INPUT_SIZE : int = 50
MLP_HIDDEN_OUTPUT_SIZE: int = 30

MLP_NUM_HIDDEN_INPUT_POSSIBILITIES : List[int] = [3, 5, 7, 9]
MLP_NUM_HIDDEN_OUTPUT_POSSIBILITIES: List[int] = [3, 5, 7, 9]


# -----------------------------------------------------------------------------
# General Optimizer, LR scheduler and Gradient Clipping Configurations
# -----------------------------------------------------------------------------
LR          : float = 0.0001
WEIGHT_DECAY: float = 1e-04
PATIENCE    : int   = 5
GRAD_CLIP   : int   = 5
MIN_LR      : float = 1.0e-6
FACTOR      : float = 0.5
MODE        : str   = "min"