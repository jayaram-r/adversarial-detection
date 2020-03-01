import os
import numpy

ROOT = '/nobackup/varun/adversarial-detection/expts'
DATA_PATH = os.path.join(ROOT, 'data')
NUMPY_DATA_PATH = os.path.join(ROOT, 'numpy_data')
MODEL_PATH = os.path.join(ROOT, 'models')
OUTPUT_PATH = os.path.join(ROOT, 'outputs')

# Normalization constants for the different datasets
NORMALIZE_IMAGES = {
    'mnist': ((0.1307,), (0.3081,)),
    'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'svhn': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
}

# Number of neighbors is calculated as a function of the data size (number of samples) `N`.
# Number of neighbors, `k = N ** NEIGHBORHOOD_CONST`.
NEIGHBORHOOD_CONST = 0.4

# Constants used by the NN-descent method
MIN_N_NEIGHBORS = 20
RHO = 0.5

SEED_DEFAULT = 123

# Default batch size
BATCH_SIZE_DEF = 128

# Default number of folds to use for cross-validation
CROSS_VAL_SIZE = 5

# Number of Gaussian noise standard deviation values to use
NUM_NOISE_VALUES = 10

# Distance metric to use if not specified
METRIC_DEF = 'cosine'

# Method to estimate intrinsic dimension
METHOD_INTRINSIC_DIM = 'lid_mle'

# Method for dimension reduction
METHOD_DIM_REDUCTION = 'NPP'
MAX_SAMPLES_DIM_REDUCTION = 10000

# Cumulative variance cutoff for PCA
PCA_CUTOFF = 0.995

# Proportion of noisy samples to include in the training or test folds of cross-validation
NOISE_PROPORTION = 0.05

# Number of top ranked layer test statistics to use for detection (default value)
NUM_TOP_RANKED = 3

# List of detection methods
DETECTION_METHODS = ['proposed', 'lid', 'odds', 'dknn', 'trust']

# Method names to use for plots and results
METHOD_NAME_MAP = {
    'proposed': 'proposed',
    'lid': 'LID_ICLR',
    'odds': 'odds_are_odd',
    'dknn': 'deep_KNN',
    'trust': 'trust_score'
}

# List of layerwise test statistics supported by the proposed method
TEST_STATS_SUPPORTED = ['multinomial', 'lid', 'lle']

# Score type used by the proposed method
SCORE_TYPES = ['density', 'pvalue']

# Layers at which the trust score calculation is supported. 'prelogit' refers to the fully connected layer
# preceding the logit layer.
LAYERS_TRUST_SCORE = ['input', 'logit', 'prelogit']

# norm type used for the different attack methods
ATTACK_NORM_MAP = {
    'FGSM': 'inf',
    'PGD': 'inf',
    'CW': '2'
}

# epsilon values used for the PGD and FGSM attacks
EPSILON_VALUES = [i / 255. for i in range(1, 21, 2)]

# Maximum FPR values for calculating partial AUC
FPR_MAX_PAUC = [0.01, 0.05, 0.1, 0.2]

# FPR threshold values for calculating TPR values.
# 0.1%, 0.5%, 1%, 5%, and 10%
FPR_THRESH = [0.001, 0.005, 0.01, 0.05, 0.1]

COLORS = ['r', 'b', 'g', 'y', 'orange', 'm', 'lawngreen', 'gold', 'c', 'hotpink', 'grey', 'steelblue',
          'tan', 'lightsalmon', 'navy']
