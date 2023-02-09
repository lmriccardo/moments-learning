from typing import List, Dict, Any
import torch.nn as nn
import torch.optim as optim
import os.path as opath
import os

# -----------------------------------------------------------------------------
# KFold Cross Validation Configurations
# -----------------------------------------------------------------------------
KF_SPLIT        : int  = 3
KF_POSSIBILITIES: List[int] = [3, 5, 7, 9]

# -----------------------------------------------------------------------------
# General Training and Testing Configurations
# -----------------------------------------------------------------------------
BATCH_SIZE        : int             = 32
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
PATIENCE    : int   = 10
GRAD_CLIP   : int   = 5
MIN_LR      : float = 1.0e-6
FACTOR      : float = 0.5
MODE        : str   = "min"

# -----------------------------------------------------------------------------
# Inverse Problem Configuration for RandomForest Regression
# -----------------------------------------------------------------------------
RAND_SEARCH_N_ESTIMATORS     : List[int]  = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
RAND_SEARCH_BOOSTRAP         : List[bool] = [True, False]
RAND_SEARCH_MAX_DEPTH        : List[int]  = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
RAND_SEARCH_MAX_FEATURES     : List[str]  = [1.0, 'sqrt']
RAND_SEARCH_MIN_SAMPLES_LEAF : List[int]  = [1, 2, 4]
RAND_SEARCH_MIN_SAMPLES_SPLIT: List[int]  = [2, 5, 10]

RAND_SEARCH_NUM_CROSS_VALIDATION : int = 3
RAND_SEARCH_NUM_ITERATIONS : int = 100

GRID_SEARCH_NUM_CROSS_VALIDATION : int = 3
GRID_SEARCH_BASE_PARAMETERS : Dict[str, Any] = {
        'n_estimators'      : 800,
        'min_samples_split' : 10,
        'min_samples_leaf'  : 4,
        'max_features'      : 'sqrt',
        'max_depth'         : 100,
        'bootstrap'         : True
}