"""
Anomaly detection based on parametric modeling of the distribution of local intrinsic dimension.
The local intrinsic dimension (LID) estimates are calculated at each nominal data point based on the maximum
likelihood estimator from [1] and [2]. The distribution of LID values from nominal data is then modeled parametrically
using a mixture of Log-Normal densities. In other words, the log of the LID values are modeled using a mixture
of Normal densities. Given a test point, its LID estimate and p-value under the parametric density model are
calculated. A very small p-value corresponds to an anomalous test point.

1. Amsaleg, Laurent, et al. "Estimating local intrinsic dimensionality." Proceedings of the 21th ACM SIGKDD
International Conference on Knowledge Discovery and Data Mining. ACM, 2015.

2. Ma, Xingjun, et al. "Characterizing adversarial subspaces using local intrinsic dimensionality."
arXiv preprint arXiv:1801.02613 (2018).

"""
import numpy as np
import sys
from pynndescent import NNDescent
from lid_estimators import lid_mle_amsaleg
from metrics_custom import remove_self_neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
import warnings
from numba import NumbaPendingDeprecationWarning

# Suppress numba warnings
warnings.filterwarnings('ignore', '', NumbaPendingDeprecationWarning)


class LID_based_anomaly_detection:
    def __init__(self,
                 neighborhood_constant=0.4, n_neighbors=None,
                 metric='euclidean', metric_kwargs=None,
                 approx_nearest_neighbors=True,
                 n_jobs=1,
                 seed_rng=123):
        """

        :param neighborhood_constant: float value in (0, 1), that specifies the number of nearest neighbors as a
                                      function of the number of samples (data size). If `N` is the number of samples,
                                      then the number of neighbors is set to `N^neighborhood_constant`. It is
                                      recommended to set this value in the range 0.4 to 0.5.
        :param n_neighbors: None or int value specifying the number of nearest neighbors. If this value is specified,
                            the `neighborhood_constant` is ignored. It is sufficient to specify either
                            `neighborhood_constant` or `n_neighbors`.
        :param metric: string or a callable that specifies the distance metric.
        :param metric_kwargs: optional keyword arguments required by the distance metric specified in the form of a
                              dictionary.
        :param approx_nearest_neighbors: Set to True in order to use an approximate nearest neighbor algorithm to
                                         find the nearest neighbors. This is recommended when the number of points is
                                         large and/or when the dimension of the data is high.
        :param n_jobs: Number of parallel jobs or processes. Set to -1 to use all the available cpu cores.
        :param seed_rng: int value specifying the seed for the random number generator.
        """
        self.neighborhood_constant = neighborhood_constant
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.approx_nearest_neighbors = approx_nearest_neighbors
        self.n_jobs = n_jobs
        self.seed_rng = seed_rng

        self.num_samples = None
        self.index_knn = None
        self.lid_nominal = None
        np.random.seed(self.seed_rng)

    def fit(self, data):
        """
        :param data: numpy data array of shape `(N, d)`, where `N` is the number of samples and `d` is the number
                     of dimensions (features).
        :return: None
        """
        N, d = data.shape
        self.num_samples = N
        if self.n_neighbors is None:
            # Set number of nearest neighbors based on the data size and the neighborhood constant
            self.n_neighbors = int(np.ceil(N ** self.neighborhood_constant))

        # Build the KNN graph for the data
        self.index_knn = self.build_knn_graph(data)

        # LID estimate at each point based on the nearest neighbor distances
        self.lid_nominal = self.estimate_lid(data, exclude_self=True)

    def score(self, data_test, exclude_self=False):
        """
        Calculate the anomaly score (p-value) for a given test data set.

        :param data_test: numpy data array of shape `(N, d)`, where `N` is the number of samples and `d` is the
                          number of dimensions (features).
        :param exclude_self: Set to True if the points in `data` were already used to build the KNN index.
        :return score: numpy array of shape `(n, 1)` containing the score for each point. Points with higher score
                       are more likely to be anomalous.
        """
        # LID estimate at each point based on the nearest neighbor distances
        lid = self.estimate_lid(data_test, exclude_self=exclude_self)

        # Estimate the p-value of each test point based on the empirical distribution of LID on the nominal data
        pvalues = ((1. / self.lid_nominal.shape[0]) *
                   np.sum(self.lid_nominal[:, np.newaxis] > lid[np.newaxis, :], axis=0))

        # Negative log of the p-value is returned as the anomaly score
        return -1.0 * np.log(np.clip(pvalues, sys.float_info.min, None))

    def estimate_lid(self, data, exclude_self=False):
        """
        :param data: numpy data array of shape `(N, d)`, where `N` is the number of samples and `d` is the number
                     of dimensions (features).
        :param exclude_self: Set to True if the points in `data` were already used to build the KNN index.
        :return:
        """
        # Find the distances from each point to its `self.n_neighbors` nearest neighbors.
        # Query an additional neighbor if the point happens to be part of the KNN graph
        k = self.n_neighbors + 1 if exclude_self else self.n_neighbors
        nn_indices_, nn_distances_ = self.query_wrapper(data, self.index_knn, k)

        if exclude_self:
            _, nn_distances = remove_self_neighbors(nn_indices_, nn_distances_)
        else:
            nn_distances = nn_distances_

        return lid_mle_amsaleg(nn_distances)

    def build_knn_graph(self, data, min_n_neighbors=20, rho=0.5):
        """
        Build a KNN index for the given data set.
        :param data: numpy data array of shape `(N, d)`, where `N` is the number of samples and `d` is the number
                     of dimensions (features).
        :param min_n_neighbors: minimum number of nearest neighbors to use for the `NN-descent` method.
        :param rho: `rho` parameter used by the `NN-descent` method.
        :return: KNN index
        """
        if self.approx_nearest_neighbors:
            # Construct an approximate nearest neighbor (ANN) index to query nearest neighbors
            params = {
                'metric': self.metric,
                'metric_kwds': self.metric_kwargs,
                'n_neighbors': max(1 + self.n_neighbors, min_n_neighbors),
                'rho': rho,
                'random_state': self.seed_rng,
                'n_jobs': self.n_jobs
            }
            index_knn = NNDescent(data, **params)
        else:
            # Construct the exact KNN graph
            index_knn = NearestNeighbors(
                n_neighbors=(1 + self.n_neighbors),
                algorithm='brute',
                metric=self.metric,
                metric_params=self.metric_kwargs,
                n_jobs=self.n_jobs
            )
            index_knn.fit(data)

        return index_knn

    def query_wrapper(self, data, index, k):
        """
        Unified wrapper for querying both the approximate and the exact KNN index.

        :param data: numpy data array of shape `(N, d)`, where `N` is the number of samples and `d` is the number
                     of dimensions (features).
        :param index: KNN index.
        :param k: number of nearest neighbors to query.
        :return:
        """
        if self.approx_nearest_neighbors:
            nn_indices, nn_distances = index.query(data, k=k)
        else:
            nn_distances, nn_indices = index.kneighbors(data, n_neighbors=k)

        return nn_indices, nn_distances
