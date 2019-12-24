"""
Class for construction of a K nearest neighbors index with support for custom distance metrics and
shared nearest neighbors (SNN) distance.

USAGE:
```
from knn_index import KNNIndex

index = KNNIndex(data, **kwargs)
nn_indices, nn_distances = index.query(data, k=5, exclude_self=True)

```
"""
import numpy as np
from pynndescent import NNDescent
from sklearn.neighbors import NearestNeighbors
from metrics_custom import (
    distance_SNN,
    remove_self_neighbors
)
import warnings
from numba import NumbaPendingDeprecationWarning

# Suppress numba warnings
warnings.filterwarnings('ignore', '', NumbaPendingDeprecationWarning)


class KNNIndex:
    """
    Class for construction of a K nearest neighbors index with support for custom distance metrics and
    shared nearest neighbors (SNN) distance.
    """
    def __init__(self, data,
                 neighborhood_constant=0.4, n_neighbors=None,
                 metric='euclidean', metric_kwargs=None,
                 shared_nearest_neighbors=False,
                 approx_nearest_neighbors=True,
                 n_jobs=1,
                 seed_rng=123):
        """
        :param data: numpy array with the data samples. Has shape `(N, d)`, where `N` is the number of samples and
                     `d` is the number of features.
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
        self.data = data
        self.neighborhood_constant = neighborhood_constant
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.shared_nearest_neighbors = shared_nearest_neighbors
        self.approx_nearest_neighbors = approx_nearest_neighbors
        self.n_jobs = n_jobs
        self.seed_rng = seed_rng

        N, d = data.shape
        if self.n_neighbors is None:
            # Set number of nearest neighbors based on the data size and the neighborhood constant
            self.n_neighbors = int(np.ceil(N ** self.neighborhood_constant))

        # Number of neighbors to use for calculating the shared nearest neighbor distance
        self.n_neighbors_snn = min(int(1.2 * self.n_neighbors), N - 1)
        # self.n_neighbors_snn = self.n_neighbors

        self.index_knn = self.build_knn_index(data)

    def build_knn_index(self, data, min_n_neighbors=20, rho=0.5):
        """
        Build a KNN index for the given data set. There will two KNN indices of the SNN distance is used.

        :param data: numpy data array of shape `(N, d)`, where `N` is the number of samples and `d` is the number
                     of dimensions (features).
        :param min_n_neighbors: minimum number of nearest neighbors to use for the `NN-descent` method.
        :param rho: `rho` parameter used by the `NN-descent` method.

        :return: A list with one or two KNN indices.
        """
        # Add one extra neighbor because querying on the points that are part of the KNN index will result in
        # the neighbor set containing the queried point. This can be removed from the query result
        if self.shared_nearest_neighbors:
            k = max(1 + self.n_neighbors_snn, min_n_neighbors)
        else:
            k = max(1 + self.n_neighbors, min_n_neighbors)

        # KNN index based on the primary distance metric
        if self.approx_nearest_neighbors:
            params = {
                'metric': self.metric,
                'metric_kwds': self.metric_kwargs,
                'n_neighbors': k,
                'rho': rho,
                'random_state': self.seed_rng,
                'n_jobs': self.n_jobs
            }
            index_knn_primary = NNDescent(data, **params)
        else:
            # Exact KNN graph
            index_knn_primary = NearestNeighbors(
                n_neighbors=k,
                algorithm='brute',
                metric=self.metric,
                metric_params=self.metric_kwargs,
                n_jobs=self.n_jobs
            )
            index_knn_primary.fit(data)

        if self.shared_nearest_neighbors:
            # Construct a second KNN index that uses the shared nearest neighbor distance
            data_neighbors, _ = remove_self_neighbors(
                *self.query_wrapper_(data, index_knn_primary, self.n_neighbors_snn + 1)
            )
            if self.approx_nearest_neighbors:
                params = {
                    'metric': distance_SNN,
                    'n_neighbors': max(1 + self.n_neighbors, min_n_neighbors),
                    'rho': rho,
                    'random_state': self.seed_rng,
                    'n_jobs': self.n_jobs
                }
                index_knn_secondary = NNDescent(data_neighbors, **params)
            else:
                index_knn_secondary = NearestNeighbors(
                    n_neighbors=(1 + self.n_neighbors),
                    algorithm='brute',
                    metric=distance_SNN,
                    n_jobs=self.n_jobs
                )
                index_knn_secondary.fit(data_neighbors)

            index_knn = [index_knn_primary, index_knn_secondary]
        else:
            index_knn = [index_knn_primary]

        return index_knn

    def query(self, data, k=None, exclude_self=False):
        """
        Query for the `k` nearest neighbors of each point in `data`.

        :param data: numpy data array of shape `(N, d)`, where `N` is the number of samples and `d` is the number
                     of dimensions (features).
        :param k: number of nearest neighbors to query. If not specified or set to `None`, `k` will be
                  set to `self.n_neighbors`.
        :param exclude_self: Set to True if `data` was used to construct the KNN index. This will ensure that points
                             are not included in their own neighborhood set.

        :return: (nn_indices, nn_distances), where
            - nn_indices: numpy array of indices of the nearest neighbors. Has shape `(data.shape[0], k)`.
            - nn_distances: numpy array of distances of the nearest neighbors. Has shape `(data.shape[0], k)`.
        """
        if k is None:
            k = self.n_neighbors

        if self.shared_nearest_neighbors:
            if exclude_self:
                data_neighbors, _ = remove_self_neighbors(
                    *self.query_wrapper_(data, self.index_knn[0], self.n_neighbors_snn + 1)
                )
                nn_indices, nn_distances = remove_self_neighbors(
                    *self.query_wrapper_(data_neighbors, self.index_knn[1], k + 1)
                )
            else:
                data_neighbors, _ = self.query_wrapper_(data, self.index_knn[0], self.n_neighbors_snn)
                nn_indices, nn_distances = self.query_wrapper_(data_neighbors, self.index_knn[1], k)

        else:
            if exclude_self:
                nn_indices, nn_distances = remove_self_neighbors(
                    *self.query_wrapper_(data, self.index_knn[0], k + 1)
                )
            else:
                nn_indices, nn_distances = self.query_wrapper_(data, self.index_knn[0], k)

        return nn_indices, nn_distances

    def query_wrapper_(self, data, index, k):
        """
        Unified wrapper for querying both the approximate and the exact KNN index.

        :param data: numpy data array of shape `(N, d)`, where `N` is the number of samples and `d` is the number
                     of dimensions (features).
        :param index: KNN index.
        :param k: number of nearest neighbors to query.

        :return: (nn_indices, nn_distances), where
            - nn_indices: numpy array of indices of the nearest neighbors. Has shape `(data.shape[0], k)`.
            - nn_distances: numpy array of distances of the nearest neighbors. Has shape `(data.shape[0], k)`.
        """
        if self.approx_nearest_neighbors:
            nn_indices, nn_distances = index.query(data, k=k)
        else:
            nn_distances, nn_indices = index.kneighbors(data, n_neighbors=k)

        return nn_indices, nn_distances
