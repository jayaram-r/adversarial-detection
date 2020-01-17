
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

# Cumulative variance cutoff for PCA
PCA_CUTOFF = 0.995
