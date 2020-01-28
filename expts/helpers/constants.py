import os

ROOT = '/nobackup/varun/adversarial-detection/expts'
DATA_PATH = os.path.join(ROOT, 'data')
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

# Default number of folds to use for cross-validation
CROSS_VAL_SIZE = 5

# Distance metric to use if not specified
METRIC_DEF = 'cosine'

# Method to estimate intrinsic dimension
METHOD_INTRINSIC_DIM = 'lid_mle'

# Method for dimension reduction
METHOD_DIM_REDUCTION = 'NPP'
MAX_SAMPLES_DIM_REDUCTION = 10000

# Cumulative variance cutoff for PCA
PCA_CUTOFF = 0.995

# Default proportion of attack samples in the test set
ATTACK_PROPORTION_DEF = 0.1

# Number of top ranked layer test statistics to use for detection (default value)
NUM_TOP_RANKED = 3
