"""
Implementation of the trust score for identifying trustworthy and suspicious predictions of a classifier.

Jiang, Heinrich, et al. "To trust or not to trust a classifier." Advances in neural information processing
systems. 2018.

We use the negative logarithm of the trust score to rank samples for out-of-distribution and adversarial detection.
We also provide the option of performing dimensionality reduction on the input feature vectors using the neighborhood
preserving projection (NPP) method.

The paper used the following values for experiments:
Number of nearest neighbors (k) = 10.
Fraction of outliers (alpha) is set to 0 in most experiments.
PCA is used to project high dimensional data sets to a dimension of 20.

"""
import numpy as np
import sys
import logging
from helpers.constants import (
    SEED_DEFAULT,
    NEIGHBORHOOD_CONST
)
from helpers.dimension_reduction_methods import (
    transform_data_from_model,
    load_dimension_reduction_models
)
from helpers.knn_index import KNNIndex
from helpers.utils import get_num_jobs

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


class TrustScore:
    """
    Implementation of the trust score for identifying trustworthy and suspicious predictions of a classifier.

    Jiang, Heinrich, et al. "To trust or not to trust a classifier." Advances in neural information processing
    systems. 2018.
    """
    _name = 'trust'
    def __init__(self,
                 alpha=0.0,     # fraction of outliers
                 neighborhood_constant=NEIGHBORHOOD_CONST, n_neighbors=None,
                 metric='euclidean', metric_kwargs=None,
                 approx_nearest_neighbors=True,
                 skip_dim_reduction=True,
                 model_dim_reduction=None,
                 n_jobs=1,
                 low_memory=False,
                 seed_rng=SEED_DEFAULT):
        """
        :param alpha: float value in (0, 1) specifying the proportion of outliers. This defines the `1 - alpha`
                      density level set.
        :param neighborhood_constant: float value in (0, 1), that specifies the number of nearest neighbors as a
                                      function of the number of samples (data size). If `N` is the number of samples,
                                      then the number of neighbors is set to `N^neighborhood_constant`. It is
                                      recommended to set this value in the range 0.4 to 0.5.
        :param n_neighbors: None or int value specifying the number of nearest neighbors. If this value is specified,
                            the `neighborhood_constant` is ignored. It is sufficient to specify either
                            `neighborhood_constant` or `n_neighbors`.
        :param metric: string or a callable that specifies the distance metric to use. We set the default metric to
                       'euclidean' since that is also used in the paper.
        :param metric_kwargs: optional keyword arguments required by the distance metric specified in the form of a
                              dictionary.
        :param approx_nearest_neighbors: Set to True in order to use an approximate nearest neighbor algorithm to
                                         find the nearest neighbors. The NN-descent method is used for approximate
                                         nearest neighbor searches.
        :param skip_dim_reduction: Set to True in order to skip dimension reduction of the layer embeddings.
        :param model_dim_reduction: None or a dimensionality reduction model object that linearly transforms
                                    (projects) data to a lower dimension. The transformation matrix can be obtained
                                    by applying linear dimensionality reduction methods such as neighborhood
                                    preserving projection (NPP) or PCA.
        :param n_jobs: Number of parallel jobs or processes. Set to -1 to use all the available cpu cores.
        :param low_memory: Set to True to enable the low memory option of the `NN-descent` method. Note that this
                           is likely to increase the running time.
        :param seed_rng: int value specifying the seed for the random number generator. This is passed around to
                         all the classes/functions that require random number generation. Set this to a fixed value
                         for reproducible results.
        """
        self.alpha = alpha
        self.neighborhood_constant = neighborhood_constant
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.approx_nearest_neighbors = approx_nearest_neighbors
        self.skip_dim_reduction = skip_dim_reduction
        self.model_dim_reduction = model_dim_reduction
        self.n_jobs = get_num_jobs(n_jobs)
        self.low_memory = low_memory
        self.seed_rng = seed_rng

        np.random.seed(self.seed_rng)
        # Check the dimension reduction model
        if skip_dim_reduction:
            self.model_dim_reduction = None
        else:
            if self.model_dim_reduction is None:
                raise ValueError("Model for dimensionality reduction is required but not specified as input.")

        self.labels_unique = None
        self.n_classes = None
        self.n_samples = None
        # KNN index for the alpha-high density samples from each class
        self.index_knn = None
        # Threshold on the k-NN radius for each class
        self.epsilon = None
        # Trust scores on the training data
        self.scores_estim = None

    def fit(self, data, labels, labels_pred):
        """
        Estimate the `1 - alpha` density level sets for each class using the given data, with true labels and
        classifier-predicted labels. This will be used to calculate the trust score.

        :param data: numpy array with the feature vectors of shape `(n, d)`, where `n` and `d` are the number of
                     samples and the data dimension respectively.
        :param labels: numpy array of labels for the classification problem addressed by the DNN. Should have shape
                       `(n, )`, where `n` is the number of samples.
        :param labels_pred: numpy array similar to `labels`, but with the classes predicted by the classifier.

        :return: Instance of the class with all parameters fit to the data.
        """
        self.n_samples, dim = data.shape
        self.labels_unique = np.unique(labels)
        self.n_classes = len(self.labels_unique)
        if self.n_neighbors is None:
            # Set number of nearest neighbors based on the maximum number of samples per class and the neighborhood
            # constant
            num = 0
            for c in self.labels_unique:
                ind = np.where(labels == c)[0]
                if ind.shape[0] > num:
                    num = ind.shape[0]

            self.n_neighbors = int(np.ceil(num ** self.neighborhood_constant))

        logger.info("Number of samples: {:d}. Data dimension = {:d}.".format(self.n_samples, dim))
        logger.info("Number of classes: {:d}.".format(self.n_classes))
        logger.info("Number of neighbors (k): {:d}.".format(self.n_neighbors))
        logger.info("Fraction of outliers (alpha): {:.4f}.".format(self.alpha))
        if self.model_dim_reduction:
            data = transform_data_from_model(data, self.model_dim_reduction)
            dim = data.shape[1]
            logger.info("Applying dimension reduction to the data. Projected dimension = {:d}.".format(dim))

        # Distance from each sample in `data` to the `1 - alpha` level sets corresponding to each class
        distance_level_sets = np.zeros((self.n_samples, self.n_classes))
        self.index_knn = dict()
        self.epsilon = dict()
        indices_sub = dict()
        for j, c in enumerate(self.labels_unique):
            logger.info("Processing data from class '{}':".format(c))
            logger.info("Building a KNN index for all the samples from class '{}'.".format(c))
            indices_sub[c] = np.where(labels == c)[0]
            data_sub = data[indices_sub[c], :]
            self.index_knn[c] = KNNIndex(
                data_sub, n_neighbors=self.n_neighbors,
                metric=self.metric, metric_kwargs=self.metric_kwargs,
                approx_nearest_neighbors=self.approx_nearest_neighbors,
                n_jobs=self.n_jobs,
                low_memory=self.low_memory,
                seed_rng=self.seed_rng
            )
            # Distances to the k nearest neighbors of each sample
            _, nn_distances = self.index_knn[c].query_self(k=self.n_neighbors)
            # Radius or distance to the k-th nearest neighbor for each sample
            radius_arr = nn_distances[:, self.n_neighbors - 1]

            # Smallest radius `epsilon` such that only `alpha` fraction of the samples from class `c` have radius
            # greater than `epsilon`
            if self.alpha > 0.:
                self.epsilon[c] = np.percentile(radius_arr, 100 * (1 - self.alpha), interpolation='midpoint')

                # Exclude the outliers and build a KNN index with the remaining samples
                mask_incl = radius_arr <= self.epsilon[c]
                mask_excl = np.logical_not(mask_incl)
                num_excl = mask_excl[mask_excl].shape[0]
            else:
                # Slightly larger value than the largest radius
                self.epsilon[c] = 1.0001 * np.max(radius_arr)

                # All samples are included in the density level set
                mask_incl = np.ones(indices_sub[c].shape[0], dtype=np.bool)
                mask_excl = np.logical_not(mask_incl)
                num_excl = 0

            if num_excl:
                logger.info("Excluding {:d} samples with radius larger than {:.6f} and building a KNN index with "
                            "the remaining samples.".format(num_excl, self.epsilon[c]))
                self.index_knn[c] = KNNIndex(
                    data_sub[mask_incl, :], n_neighbors=self.n_neighbors,
                    metric=self.metric, metric_kwargs=self.metric_kwargs,
                    approx_nearest_neighbors=self.approx_nearest_neighbors,
                    n_jobs=self.n_jobs,
                    low_memory=self.low_memory,
                    seed_rng=self.seed_rng
                )
                # Distance to the nearest neighbor of each sample that is part of the KNN index
                _, dist_temp = self.index_knn[c].query_self(k=1)
                ind = indices_sub[c][mask_incl]
                distance_level_sets[ind, j] = dist_temp[:, 0]

                # Distance to the nearest neighbor of each sample that is not a part of the KNN index (outliers)
                _, dist_temp = self.index_knn[c].query(data_sub[mask_excl, :], k=1)
                ind = indices_sub[c][mask_excl]
                distance_level_sets[ind, j] = dist_temp[:, 0]
            else:
                # No need to rebuild the KNN index because no samples are excluded.
                # Distance to the nearest neighbor of each sample
                distance_level_sets[indices_sub[c], j] = nn_distances[:, 0]

        logger.info("Calculating the trust score for the estimation data.")
        for c in self.labels_unique:
            # Compute the distance from each sample from class `c` to the level sets from the remaining classes
            data_sub = data[indices_sub[c], :]
            for j, c_hat in enumerate(self.labels_unique):
                if c_hat == c:
                    continue

                _, dist_temp = self.index_knn[c_hat].query(data_sub, k=1)
                distance_level_sets[indices_sub[c], j] = dist_temp[:, 0]

        self.scores_estim = self._score_helper(distance_level_sets, labels_pred)
        return self

    def score(self, data_test, labels_pred, is_train=False):
        """
        Calculate the score for detecting samples that are not trust-worthy, such as out-of-distribution and
        adversarial samples. This is the negative log of the trust score. Hence, a large value of this score should
        (ideally) correspond to a high probability of the sample being not trust-worthy.

        :param data: numpy array with the test data of shape `(n, d)`, where `n` and `d` are the number of samples
                     and the dimension respectively.
        :param labels_pred: numpy array of the classifier-predicted labels for the samples in `data_test`. Should
                            have shape `(n, )`.
        :param is_train: Set to true if this data was used to also passed to the `fit` method for estimation.

        :return: numpy array of scores corresponding to OOD or adversarial detection.
        """
        return -np.log(np.clip(self.score_trust(data_test, labels_pred, is_train=is_train),
                               sys.float_info.min, None))

    def score_trust(self, data_test, labels_pred, is_train=False):
        """
        Calculate the trust score of each test sample given its classifier-predicted labels. The score is non-negative,
        with higher values corresponding to a higher level of trust in the classifier's prediction.

        :param data: numpy array with the test data of shape `(n, d)`, where `n` and `d` are the number of samples
                     and the dimension respectively.
        :param labels_pred: numpy array of the classifier-predicted labels for the samples in `data_test`. Should
                            have shape `(n, )`.
        :param is_train: Set to true if this data was used to also passed to the `fit` method for estimation.

        :return: numpy array of trust scores for each test sample.
        """
        if is_train:
            return self.scores_estim

        if self.model_dim_reduction:
            data_test = transform_data_from_model(data_test, self.model_dim_reduction)

        distance_level_sets = np.zeros((data_test.shape[0], self.n_classes))
        for j, c in enumerate(self.labels_unique):
            # Distance of each test sample to its nearest neighbor from the level set for class `c`
            _, dist_temp = self.index_knn[c].query(data_test, k=1)
            distance_level_sets[:, j] = dist_temp[:, 0]

        # Trust score calculation
        return self._score_helper(distance_level_sets, labels_pred)

    def _score_helper(self, distance_level_sets, labels_pred):
        # A helper function to calculate the trust score from the distances of samples to the level sets
        scores = np.zeros(distance_level_sets.shape[0])
        mask = np.ones((self.n_classes, self.n_classes), dtype=np.bool)
        np.fill_diagonal(mask, False)
        for j, c in enumerate(self.labels_unique):
            # All samples predicted into class `c`
            ind = np.where(labels_pred == c)[0]
            if ind.shape[0] == 0:
                continue

            dist_temp = distance_level_sets[ind, :]
            # Minimum distance to the level sets from classes other than `c` is the numerator of the trust score.
            # Distance to the level set of the predicted class `c` is the denominator of the trust score.
            scores[ind] = np.min(dist_temp[:, mask[j, :]], axis=1) / np.clip(dist_temp[:, j], 1e-32, None)

        return scores
