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
from lid_estimators import lid_mle_amsaleg
from knn_index import KNNIndex
from utils import get_num_jobs


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
        self.n_jobs = get_num_jobs(n_jobs)
        self.seed_rng = seed_rng

        self.num_samples = None
        self.index_knn = None
        self.lid_nominal = None

    def fit(self, data):
        """
        :param data: numpy data array of shape `(N, d)`, where `N` is the number of samples and `d` is the number
                     of dimensions (features).
        :return: None
        """
        self.num_samples = data.shape[0]
        # Build the KNN graph for the data
        self.index_knn = KNNIndex(
            data,
            neighborhood_constant=self.neighborhood_constant, n_neighbors=self.n_neighbors,
            metric=self.metric, metric_kwargs=self.metric_kwargs,
            shared_nearest_neighbors=False,
            approx_nearest_neighbors=self.approx_nearest_neighbors,
            n_jobs=self.n_jobs,
            seed_rng=self.seed_rng
        )
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
        nn_indices, nn_distances = self.index_knn.query(data, exclude_self=exclude_self)

        return lid_mle_amsaleg(nn_distances)
