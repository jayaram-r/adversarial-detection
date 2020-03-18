"""
Methods for (local) intrinsic dimension estimation based on nearest neighbor distances.

"""
import numpy as np
from scipy import stats
import sys
from helpers.knn_index import KNNIndex
from helpers.constants import (
    NEIGHBORHOOD_CONST,
    SEED_DEFAULT
)


def lid_mle_amsaleg(knn_distances):
    """
    Local intrinsic dimension (LID) estimators from the papers,
    1. Amsaleg, Laurent, et al. "Estimating local intrinsic dimensionality." Proceedings of the 21th ACM SIGKDD
    International Conference on Knowledge Discovery and Data Mining. ACM, 2015.

    2. Ma, Xingjun, et al. "Characterizing adversarial subspaces using local intrinsic dimensionality."
    arXiv preprint arXiv:1801.02613 (2018).

    :param knn_distances: numpy array of k nearest neighbor distances. Has shape `(n, k)` where `n` is the
                          number of points and `k` is the number of neighbors.
    :return: `lid_est` is a numpy array of shape `(n, )` with the local intrinsic dimension estimates
              in the neighborhood of each point.
    """
    n, k = knn_distances.shape
    # Replace 0 distances with a very small float value
    knn_distances = np.clip(knn_distances, sys.float_info.min, None)
    log_dist_ratio = np.log(knn_distances) - np.log(knn_distances[:, -1].reshape((n, 1)))
    # lid_est = -k / np.sum(log_dist_ratio, axis=1)
    lid_est = -(k - 1) / np.sum(log_dist_ratio, axis=1)

    return lid_est


def id_two_nearest_neighbors(knn_distances):
    """
    Estimate the intrinsic dimension of the data using the Two-nearest-neighbor method proposed in the following
    paper:
    Facco, Elena, et al. "Estimating the intrinsic dimension of datasets by a minimal neighborhood information."
    Scientific reports 7.1 (2017): 12140.

    :param knn_distances: numpy array of k nearest neighbor distances. Has shape `(n, k)` where `n` is the
                          number of points and `k` is the number of neighbors.
    :return: float value estimate of the intrinsic dimension.
    """
    # Ratio of 2nd to 1st nearest neighbor distances. Defined only if the 2nd nearest neighbor distance is > 0.
    mask = knn_distances[:, 1] > 0.
    d2 = knn_distances[mask, 1]
    d1 = knn_distances[mask, 0]
    n = d1.shape[0]
    log_nn_ratio = np.log(d2) - np.log(np.clip(d1, sys.float_info.min, None))

    # Empirical CDF of `log_nn_ratio`
    log_nn_ratio_sorted = np.sort(log_nn_ratio)
    # Insert value `log(1) = 0` as the minimum, which will have an empirical CDF of 0.
    # This will ensure that the line passes through the origin
    log_nn_ratio_sorted = np.insert(log_nn_ratio_sorted, 0, 0.)
    ecdf = np.arange(n + 1) / float(n + 1)

    xs = log_nn_ratio_sorted
    ys = -1 * np.log(1 - ecdf)
    # Fit a straight line. The slope of this line gives an estimate of the intrinsic dimension
    slope, intercept, _, _, _ = stats.linregress(xs, ys)

    return slope


def estimate_intrinsic_dimension(data,
                                 method='two_nn',       # method choices are {'two_nn', 'lid_mle'}
                                 neighborhood_constant=NEIGHBORHOOD_CONST, n_neighbors=None,
                                 metric='euclidean',
                                 metric_kwargs=None,
                                 approx_nearest_neighbors=True,
                                 n_jobs=1,
                                 low_memory=False,
                                 seed_rng=SEED_DEFAULT):
    """
    Wrapper function for estimating the intrinsic dimension of the data.

    :param data: data array of shape `(N, d)`, where `N` is the number of samples and `d` is the number of features.
    :param method: method string. Valid choices are 'two_nn' and 'lid_mle'.
    :param neighborhood_constant: float value in (0, 1), that specifies the number of nearest neighbors as a function
                                  of the number of samples (data size). If `N` is the number of samples, then the
                                  number of neighbors is set to `N^neighborhood_constant`. It is recommended to set
                                  this value in the range 0.4 to 0.5.
    :param n_neighbors: None or int value specifying the number of nearest neighbors. If this value is specified,
                        the `neighborhood_constant` is ignored. It is sufficient to specify either
                        `neighborhood_constant` or `n_neighbors`.
    :param metric: distance metric to use. Euclidean by default.
    :param metric_kwargs: optional keyword arguments for the distance metric specified as a dict.
    :param approx_nearest_neighbors: Set to True to use an approximate nearest neighbor method. Usually the right
                                     choice unless both the number of samples are features are small.
    :param n_jobs: number of CPU cores to use.
    :param low_memory: Set to True to enable the low memory option of the `NN-descent` method. Note that this
                       is likely to increase the running time.
    :param seed_rng: seed for the random number generator.

    :return: positive float value specifying the estimated intrinsic dimension.
    """
    # Build a KNN graph index
    index_knn = KNNIndex(data,
                         neighborhood_constant=neighborhood_constant,
                         n_neighbors=n_neighbors,
                         metric=metric, metric_kwargs=metric_kwargs,
                         shared_nearest_neighbors=False,
                         approx_nearest_neighbors=approx_nearest_neighbors,
                         n_jobs=n_jobs,
                         low_memory=low_memory,
                         seed_rng=seed_rng)
    # Query the nearest neighbors of each point
    nn_indices, nn_distances = index_knn.query_self()

    method = method.lower()
    if method == 'two_nn':
        # Two nearest neighbors ID estimator
        id = id_two_nearest_neighbors(nn_distances)
    elif method == 'lid_mle':
        # Median of the local intrinsic dimension estimates around each point
        id = np.median(lid_mle_amsaleg(nn_distances))
    else:
        raise ValueError("Invalid value '{}' specified for argument 'method'".format(method))

    return id
