
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
