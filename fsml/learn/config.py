from typing import List, Dict
import torch.nn as nn
import torch.optim as optim
import os.path as opath
import os 

# -----------------------------------------------------------------------------
# KFold Cross Validation Configurations
# -----------------------------------------------------------------------------
KF_SPLIT        : int  = 5
KF_POSSIBILITIES: List[int] = [2, 3, 5, 7]

# -----------------------------------------------------------------------------
# General Training and Testing Configurations
# -----------------------------------------------------------------------------
BATCH_SIZE        : int             = 20
NUM_EPOCHS        : int             = 200
ACCURACY_THRESHOLD: float           = 0.94
IMG_PATH          : str             = opath.join(os.getcwd(), "img")
MODEL_PATH        : str             = opath.join(os.getcwd(), "models")
OPTIMIZER         : optim.Optimizer = optim.Adam
CRITERION         : nn.Module       = nn.MSELoss
MODEL_TYPE        : str             = "MLP"
DATA_PATH         : str             = os.path.join(os.getcwd(), "data/meanstd/")


# -----------------------------------------------------------------------------
# General MLP predictor configuration and possibilities
# -----------------------------------------------------------------------------
MLP_NUM_HIDDEN_INPUT  : int = 5
MLP_NUM_HIDDEN_OUTPUT : int = 3
MLP_HIDDEN_INPUT_SIZE : int = 50
MLP_HIDDEN_OUTPUT_SIZE: int = 30

MLP_NUM_HIDDEN_INPUT_POSSIBILITIES : List[int] = [1, 3, 5, 7]
MLP_NUM_HIDDEN_OUTPUT_POSSIBILITIES: List[int] = [1, 3, 5, 7]


# -----------------------------------------------------------------------------
# General RBF predictor configuration and possibilities
# -----------------------------------------------------------------------------
RBF_NUM_HIDDEN_LAYERS: int       = 3
RBF_HIDDEN_SIZES     : List[int] = [10, 20, 10]
RBF_BASIS_FUNCTION   : str       = "gaussian"

RBF_COMBINATIONS: Dict[int, List[List[int]]] = {
    1 : [[10], [30], [50]],
    3 : [[10, 20, 10], [10, 20, 30], [20, 30, 50], [20, 30, 20]],
    5 : [[20, 30, 10, 5, 2], [10, 20, 30, 20, 10], [20, 20, 20, 30, 50]]
}


# -----------------------------------------------------------------------------
# General Optimizer, LR scheduler and Gradient Clipping Configurations
# -----------------------------------------------------------------------------
LR          : float = 0.0001
WEIGHT_DECAY: float = 1e-03
PATIENCE    : int   = 35
GRAD_CLIP   : int   = 5
MIN_LR      : float = 1.0e-6
FACTOR      : float = 0.5
MODE        : str   = "min"