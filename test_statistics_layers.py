"""
Test statistics to be calculated at the different layers of the trained deep neural network.

"""
import numpy as np
from abc import ABC, abstractmethod
import multiprocessing
from functools import partial
import logging
from knn_index import KNNIndex
from utils import get_num_jobs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestStatistic(ABC):
    """
    Skeleton class for different potential test statistics.
    """
    def __init__(self,
                 neighborhood_constant=0.4, n_neighbors=None,
                 metric='euclidean', metric_kwargs=None,
                 shared_nearest_neighbors=False,
                 approx_nearest_neighbors=True,
                 n_jobs=1,
                 seed_rng=123):

        super(TestStatistic, self).__init__()
        self.neighborhood_constant = neighborhood_constant
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.shared_nearest_neighbors = shared_nearest_neighbors
        self.approx_nearest_neighbors = approx_nearest_neighbors
        self.n_jobs = get_num_jobs(n_jobs)
        self.seed_rng = seed_rng

        self.dim = None
        self.n_train = None
        self.labels_unique = None
        self.index_knn = None
        np.random.seed(self.seed_rng)

    @abstractmethod
    def fit(self, features, labels, labels_pred, labels_unique=None):
        self.n_train, self.dim = features.shape
        if labels_unique is None:
            self.labels_unique = np.unique(labels)
        else:
            self.labels_unique = labels_unique

        if self.n_neighbors is None:
            # Set number of nearest neighbors based on the data size and the neighborhood constant
            self.n_neighbors = int(np.ceil(self.n_train ** self.neighborhood_constant))

        # Build the KNN graph
        self.index_knn = KNNIndex(
            data, n_neighbors=self.n_neighbors,
            metric=self.metric, metric_kwargs=self.metric_kwargs,
            shared_nearest_neighbors=self.shared_nearest_neighbors,
            approx_nearest_neighbors=self.approx_nearest_neighbors,
            n_jobs=self.n_jobs,
            seed_rng=self.seed_rng
        )

    @abstractmethod
    def score(self, features_test, labels_pred_test):
        pass
