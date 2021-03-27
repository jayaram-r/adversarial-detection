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
NORMALIZE_IMAGES['cifar10aug'] = NORMALIZE_IMAGES['cifar10']

# Acronym for the proposed method
METHOD_NAME_PROPOSED = 'JTLA'
# METHOD_NAME_PROPOSED = 'ReBeL'

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

# Maximum Gaussian noise standard deviation. Found using the script `generate_noisy_data.py`
NOISE_STDEV_MAX = {
    'mnist': 0.209405,
    'cifar10': 0.034199,
    'svhn': 0.038040
}
NOISE_STDEV_MAX['cifar10aug'] = NOISE_STDEV_MAX['cifar10']
# Minimum noise standard deviation values. Used to generate a range of noise standard deviations
NOISE_STDEV_MIN = {k: NOISE_STDEV_MAX[k] / 16. for k in NOISE_STDEV_MAX.keys()}

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
DETECTION_METHODS = ['mahalanobis', 'proposed', 'lid', 'lid_class_cond', 'odds', 'dknn', 'trust']

# Method names to use for plots and results
METHOD_NAME_MAP = {
    'proposed': 'proposed',
    'lid': 'LID_ICLR',
    'lid_class_cond': 'LID_ICLR_class',
    'odds': 'odds_are_odd',
    'dknn': 'deep_KNN',
    'trust': 'trust_score',
    'mahalanobis': 'deep_mahalanobis'
}

# List of layerwise test statistics supported by the proposed method
TEST_STATS_SUPPORTED = ['multinomial', 'binomial', 'lid', 'lle', 'distance', 'trust']

# Score type used by the proposed method
SCORE_TYPES = ['density', 'pvalue', 'klpe']

# Layers at which the trust score calculation is supported. 'prelogit' refers to the fully connected layer
# preceding the logit layer.
LAYERS_TRUST_SCORE = ['input', 'logit', 'prelogit']

CUSTOM_ATTACK = 'Custom'

# Maximum number of representative samples used by the custom attack. Limiting this size speeds up the custom attack
# generation and uses less memory
MAX_NUM_REPS = 2000

# norm type used for the different attack methods
ATTACK_NORM_MAP = {
    'FGSM': 'inf',
    'PGD': 'inf',
    'CW': '2',
    CUSTOM_ATTACK: '2'
}

# epsilon values used for the PGD and FGSM attacks
EPSILON_VALUES = [i / 255. for i in range(1, 21, 2)]

# Maximum FPR values for calculating partial AUC
FPR_MAX_PAUC = [0.01, 0.05, 0.1, 0.2]

# FPR threshold values for calculating TPR values.
# 0.1%, 0.5%, 1%, 5%, and 10%
FPR_THRESH = [0.001, 0.005, 0.01, 0.05, 0.1]

# Number of random samples used to estimate the p-value by the density method
NUM_RANDOM_SAMPLES = 20000

# Number of bootstrap resamples
NUM_BOOTSTRAP = 100

# Plot colors and markers
# https://matplotlib.org/2.0.2/examples/color/named_colors.html
COLORS = ['r', 'b', 'c', 'orange', 'g', 'm', 'lawngreen', 'grey', 'hotpink', 'y', 'steelblue', 'tan',
          'lightsalmon', 'navy', 'gold']
MARKERS = ['o', '^', 'v', 's', '*', 'x', 'd', '>', '<', '1', 'h', 'P', '_', '2', '|', '3', '4']
