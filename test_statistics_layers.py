"""
Test statistics to be calculated at the different layers of the trained deep neural network.

"""
import numpy as np
from abc import ABC, abstractmethod
import multiprocessing
from functools import partial
import logging
from knn_index import KNNIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestStatistic(ABC):
    """
    Skeleton class different potential test statistics.
    """
    def __init__(self,
                 neighborhood_constant=0.4, n_neighbors=None,
                 metric='euclidean', metric_kwargs=None,
                 shared_nearest_neighbors=False,
                 approx_nearest_neighbors=True,
                 n_jobs=1,
                 seed_rng=123):
        self.neighborhood_constant = neighborhood_constant
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.shared_nearest_neighbors = shared_nearest_neighbors
        self.approx_nearest_neighbors = approx_nearest_neighbors
        self.n_jobs = n_jobs
        self.seed_rng = seed_rng

        super(TestStatistic, self).__init__()

    @abstractmethod
    def fit(self, features, labels, labels_pred):
        pass

    @abstractmethod
    def score(self, features_test, labels_pred_test):
        pass
