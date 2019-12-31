"""
Localized p-value estimation methods for anomaly detection.

1. K-LPE method:
Zhao, Manqi, and Venkatesh Saligrama. "Anomaly detection with score functions based on nearest neighbor graphs."
Advances in neural information processing systems. 2009.

2. Averaged K-LPE method:
Qian, Jing, and Venkatesh Saligrama. "New statistic in p-value estimation for anomaly detection."
IEEE Statistical Signal Processing Workshop (SSP). IEEE, 2012.

"""
import numpy as np
import sys
import multiprocessing
from functools import partial
from knn_index import KNNIndex
from utils import get_num_jobs
from sklearn.metrics import pairwise_distances


def helper_distance(data1, data2, nn_indices, metric, metric_kwargs, k, i):
    if metric_kwargs is None:
        pd = pairwise_distances(
            data1[i, :].reshape(1, -1),
            Y=data2[nn_indices[i, :], :],
            metric=metric,
            n_jobs=1
        )
    else:
        pd = pairwise_distances(
            data1[i, :].reshape(1, -1),
            Y=data2[nn_indices[i, :], :],
            metric=metric,
            n_jobs=1,
            **metric_kwargs
        )

    # Sort the distances and compute the mean starting from the k-th distance
    pd = np.sort(pd[0, :])
    return np.mean(pd[(k - 1):])


class averaged_KLPE_anomaly_detection:
    def __init__(self,
                 neighborhood_constant=0.4, n_neighbors=None,
                 metric='euclidean', metric_kwargs=None,
                 shared_nearest_neighbors=False,
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
        :param shared_nearest_neighbors: Set to True in order to use the shared nearest neighbor (SNN) distance.
                                         This is a secondary distance metric that is found to be better suited to
                                         high dimensional data.
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
        self.shared_nearest_neighbors = shared_nearest_neighbors
        self.approx_nearest_neighbors = approx_nearest_neighbors
        self.n_jobs = get_num_jobs(n_jobs)
        self.seed_rng = seed_rng

        self.data_train = None
        self.neighborhood_range = None
        self.index_knn = None
        self.dist_stat_nominal = None
        np.random.seed(self.seed_rng)

    def fit(self, data):
        """
        :param data: numpy data array of shape `(N, d)`, where `N` is the number of samples and `d` is the number
                     of dimensions (features).
        :return: None
        """
        N, d = data.shape
        self.data_train = data
        if self.n_neighbors is None:
            # Set number of nearest neighbors based on the data size and the neighborhood constant
            self.n_neighbors = int(np.ceil(N ** self.neighborhood_constant))

        # The distance statistic is averaged over this neighborhood range
        low = self.n_neighbors - int(np.floor(0.5 * (self.n_neighbors - 1)))
        high = self.n_neighbors + int(np.floor(0.5 * self.n_neighbors))
        self.neighborhood_range = (low, high)

        # Build the KNN graphs
        self.index_knn = KNNIndex(
            data, n_neighbors=self.neighborhood_range[1],
            metric=self.metric, metric_kwargs=self.metric_kwargs,
            shared_nearest_neighbors=self.shared_nearest_neighbors,
            approx_nearest_neighbors=self.approx_nearest_neighbors,
            n_jobs=self.n_jobs,
            seed_rng=self.seed_rng
        )
        # Compute the distance statistic for every data point
        self.dist_stat_nominal = self.distance_statistic(data, exclude_self=True)

    def score(self, data_test, exclude_self=False):
        """
        Calculate the anomaly score (p-value) for a given test data set.

        :param data_test: numpy data array of shape `(N, d)`, where `N` is the number of samples and `d` is the
                          number of dimensions (features).
        :param exclude_self: Set to True if the points in `data` were already used to build the KNN index.
        :return score: numpy array of shape `(n, 1)` containing the score for each point. Points with higher score
                       are more likely to be anomalous.
        """
        dist_stat_test = self.distance_statistic(data_test, exclude_self=exclude_self)
        score = ((1. / self.dist_stat_nominal.shape[0]) *
                 np.sum(dist_stat_test[:, np.newaxis] <= self.dist_stat_nominal[np.newaxis, :], axis=1))

        # Negative log of the p-value is returned as the anomaly score
        return -1.0 * np.log(np.clip(score, sys.float_info.min, None))

    def distance_statistic(self, data, exclude_self=False):
        """
        Calculate the average distance statistic by querying the nearest neighbors of the given set of points.

        :param data: numpy data array of shape `(N, d)`, where `N` is the number of samples and `d` is the number
                     of dimensions (features).
        :param exclude_self: Set to True if the points in `data` were already used to build the KNN index.
        :return dist_stat: numpy array of distance statistic for each point.
        """
        nn_indices, nn_distances = self.index_knn.query(data, k=self.neighborhood_range[1],
                                                        exclude_self=exclude_self)
        if self.shared_nearest_neighbors:
            # The distance statistic is calculated based on the primary distance metric, but within the
            # neighborhood set found using the SNN distance. The idea is that for high-dimensional data,
            # the neighborhood found using SNN is more reliable
            dist_stat = self.distance_statistic_local(data, nn_indices, self.neighborhood_range[0])
        else:
            dist_stat = np.mean(nn_distances[:, (self.neighborhood_range[0] - 1):], axis=1)

        return dist_stat

    def distance_statistic_local(self, data, nn_indices, k):
        """
        Computes the mean distance statistic for each row of `data` within a local neighborhood specified by
        `nn_indices`.

        :param data: numpy data array of shape `(N, d)`, where `N` is the number of samples and `d` is the number
                     of dimensions (features).
        :param nn_indices: numpy array of `p` nearest neighbor indices with shape `(N, p)`.
        :param k: start index of the neighbor from which the mean distance is computed.
        :return dist_array: numpy array of shape `(N, )` with the mean distance values.
        """
        n = data.shape[0]
        if self.n_jobs == 1:
            dist_stat = [helper_distance(data, self.data_train, nn_indices, self.metric, self.metric_kwargs, k, i)
                         for i in range(n)]
        else:
            helper_distance_partial = partial(helper_distance, data, self.data_train, nn_indices, self.metric,
                                              self.metric_kwargs, k)
            pool_obj = multiprocessing.Pool(processes=self.n_jobs)
            dist_stat = []
            _ = pool_obj.map_async(helper_distance_partial, range(n), callback=dist_stat.extend)
            pool_obj.close()
            pool_obj.join()

        return np.array(dist_stat)
