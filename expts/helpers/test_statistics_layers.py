"""
Test statistics to be calculated at the different layers of the trained deep neural network.

"""
import numpy as np
from abc import ABC, abstractmethod
import logging
from helpers.knn_index import KNNIndex
from helpers.knn_classifier import neighbors_label_counts
from helpers.multinomial import (
    multinomial_estimation,
    special_dirichlet_prior
)
from helpers.lid_estimators import lid_mle_amsaleg
from helpers.utils import get_num_jobs
from constants import (
    NEIGHBORHOOD_CONST,
    SEED_DEFAULT,
    METRIC_DEF
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestStatistic(ABC):
    """
    Skeleton class for different potential test statistics using the DNN layer representations.
    """
    def __init__(self,
                 neighborhood_constant=NEIGHBORHOOD_CONST, n_neighbors=None,
                 metric=METRIC_DEF, metric_kwargs=None,
                 shared_nearest_neighbors=False,
                 approx_nearest_neighbors=True,
                 n_jobs=1,
                 low_memory=False,
                 seed_rng=SEED_DEFAULT):
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
        :param low_memory: Set to True to enable the low memory option of the `NN-descent` method. Note that this
                           is likely to increase the running time.
        :param seed_rng: int value specifying the seed for the random number generator.
        """
        super(TestStatistic, self).__init__()
        self.neighborhood_constant = neighborhood_constant
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.shared_nearest_neighbors = shared_nearest_neighbors
        self.approx_nearest_neighbors = approx_nearest_neighbors
        self.n_jobs = get_num_jobs(n_jobs)
        self.low_memory = low_memory
        self.seed_rng = seed_rng

        self.dim = None
        self.n_train = None
        self.labels_unique = None
        self.n_classes = None
        self.label_encoder = None
        self.index_knn = None
        np.random.seed(self.seed_rng)

    @abstractmethod
    def fit(self, features, labels, labels_pred, labels_unique=None):
        """
        Use the given feature vectors, true labels, and predicted labels to estimate the scoring model.

        :param features: numpy array of shape `(N, d)` where `N` and `d` are the number of samples and
                         dimension respectively.
        :param labels: numpy array of shape `(N, )` with the true labels per sample.
        :param labels_pred: numpy array of shape `(N, )` with the predicted labels per sample.
        :param labels_unique: None or a numpy array with the unique labels. For example, np.arange(1, 11). This can
                              be supplied as input during repeated calls to avoid having the find the unique
                              labels each time.
        :return:
        """
        self.n_train, self.dim = features.shape
        if labels_unique is None:
            self.labels_unique = np.unique(labels)
        else:
            self.labels_unique = labels_unique

        self.n_classes = len(self.labels_unique)
        # Mapping from the original labels to the set {0, 1, . . .,self.n_classes - 1}. This is needed by the label
        # count function
        d = dict(zip(self.labels_unique, np.arange(self.n_classes)))
        self.label_encoder = np.vectorize(d.__getitem__)

        # Number of nearest neighbors
        if self.n_neighbors is None:
            # Set number of nearest neighbors based on the data size and the neighborhood constant
            self.n_neighbors = int(np.ceil(self.n_train ** self.neighborhood_constant))

    @abstractmethod
    def score(self, features_test, labels_pred_test, is_train=False):
        """
        Given the feature vector and predicted labels for `N` samples, calculate their test statistic scores.

        :param features_test: numpy array of shape `(N, d)` where `N` and `d` are the number of samples and
                              dimension respectively.
        :param labels_pred_test: numpy array of shape `(N, )` with the predicted labels per sample.
        :param  is_train: Set to True if points from `features_test` were used for training, i.e. by the fit method.
                          This is used to remove points from their own set of nearest neighbors.
        :return:
        """
        pass


class MultinomialScore(TestStatistic):
    """
    Class that calculates the multinomial likelihood ratio test score from the class counts observed in the `k`
    nearest neighbors of a sample. This score is conditional on the predicted class of the sample.
    """
    def __init__(self, **kwargs):
        super(MultinomialScore, self).__init__(
            neighborhood_constant=kwargs.get('neighborhood_constant', NEIGHBORHOOD_CONST),
            n_neighbors=kwargs.get('n_neighbors', None),
            metric=kwargs.get('metric', METRIC_DEF),
            metric_kwargs=kwargs.get('metric_kwargs', None),
            shared_nearest_neighbors=kwargs.get('shared_nearest_neighbors', False),
            approx_nearest_neighbors=kwargs.get('approx_nearest_neighbors', True),
            n_jobs=kwargs.get('n_jobs', 1),
            low_memory=kwargs.get('low_memory', False),
            seed_rng=kwargs.get('seed_rng', SEED_DEFAULT)
        )
        # Encoded labels of train data
        self.labels_train_enc = None
        # Number of points from each class among the k nearest neighbors of the training data
        self.data_counts_train = None
        # Multinomial probabilities conditioned on the predicted class
        self.proba_params_pred = None
        # Multinomial probabilities conditioned on true class
        self.proba_params_true = None
        # Likelihood ratio statistic scores for the training data
        self.scores_train = None

    def fit(self, features, labels, labels_pred, labels_unique=None):
        """
        Use the given feature vectors, true labels, and predicted labels to estimate the scoring model.

        :param features: numpy array of shape `(N, d)` where `N` and `d` are the number of samples and
                         dimension respectively.
        :param labels: numpy array of shape `(N, )` with the true labels per sample.
        :param labels_pred: numpy array of shape `(N, )` with the predicted labels per sample.
        :param labels_unique: None or a numpy array with the unique labels. For example, np.arange(1, 11). This can
                              be supplied as input during repeated calls to avoid having the find the unique
                              labels each time.
        :return:
            scores_train: numpy array of two scores for each training sample. Has shape `(N, 2)`, where the first
                          column gives the scores conditioned on the predicted class, and the second column gives
                          the scores conditioned on the true class.
        """
        super(MultinomialScore, self).fit(features, labels, labels_pred, labels_unique=labels_unique)

        # Build the KNN index for the given feature vectors
        self.index_knn = KNNIndex(
            features, n_neighbors=self.n_neighbors,
            metric=self.metric, metric_kwargs=self.metric_kwargs,
            shared_nearest_neighbors=self.shared_nearest_neighbors,
            approx_nearest_neighbors=self.approx_nearest_neighbors,
            n_jobs=self.n_jobs,
            low_memory=self.low_memory,
            seed_rng=self.seed_rng
        )
        # Indices of the nearest neighbors of the points (rows) from `features`
        nn_indices, _ = self.index_knn.query_self(k=self.n_neighbors)

        # Get the class label counts from the nearest neighbors of each sample
        self.labels_train_enc = self.label_encoder(labels)
        _, self.data_counts_train = neighbors_label_counts(nn_indices, self.labels_train_enc, self.n_classes)

        # First score is conditioned the predicted class and the second score is conditioned on the true class
        self.scores_train = np.zeros(self.n_train, 2)

        # Dirichlet prior counts for MAP estimation
        alpha_diric = special_dirichlet_prior(self.n_classes)

        # Parameter estimation conditioned on the predicted class
        mat_ones = np.ones((self.n_classes, self.n_classes))
        self.proba_params_pred = (1. / self.n_classes) * mat_ones
        for c_hat in self.labels_unique:
            i = self.label_encoder([c_hat])[0]
            # Index of samples predicted into class `c_hat`
            ind = np.where(labels_pred == c_hat)[0]
            if ind.shape[0]:
                data_counts_temp = self.data_counts_train[ind, :]

                # Estimate the multinomial probability parameters given the predicted class `c_hat`
                self.proba_params_pred[i, :] = multinomial_estimation(
                    data_counts_temp, alpha_prior=alpha_diric[i, :]
                )
                # Likelihood ratio statistic for multinomial distribution
                self.scores_train[ind, 0] = self.multinomial_lrt(data_counts_temp, self.proba_params_pred[i, :],
                                                                 self.n_neighbors)
            else:
                logger.warning("No samples are predicted into class '{}'. Skipping multinomial parameter "
                               "estimation and assigning uniform probabilities.".format(c_hat))

        # Parameter estimation conditioned on the true class
        self.proba_params_true = (1. / self.n_classes) * mat_ones
        for c in self.labels_unique:
            i = self.label_encoder([c])[0]
            # Index of samples with class label `c`
            ind = np.where(labels == c)[0]
            if ind.shape[0]:
                data_counts_temp = self.data_counts_train[ind, :]

                # Estimate the multinomial probability parameters given the true class `c`
                self.proba_params_true[i, :] = multinomial_estimation(data_counts_temp,
                                                                      alpha_prior=alpha_diric[i, :])
                # Likelihood ratio statistic for multinomial distribution
                self.scores_train[ind, 1] = self.multinomial_lrt(data_counts_temp, self.proba_params_true[i, :],
                                                                 self.n_neighbors)
            else:
                # Unexpected, should not occur in practice
                logger.warning("No samples are available from class '{}'. Skipping multinomial parameter estimation "
                               "and assigning uniform probabilities.".format(c))

        return self.scores_train

    def score(self, features_test, labels_pred_test, is_train=False):
        """
        Given the test feature vectors and their corresponding predicted labels, calculate a vector of scores for
        each test sample. Set `is_train = True` only if the `fit` method was called using `features_test`.

        :param features_test: numpy array of shape `(N, d)` where `N` and `d` are the number of samples and
                              dimension respectively.
        :param labels_pred_test: numpy array of shape `(N, )` with the predicted labels per sample.
        :param  is_train: Set to True if points from `features_test` were used for training, i.e. by the fit method.

        :return: numpy array of shape `(N, m + 1)` with a vector of scores for each sample, where `m` is the number
                 of classes. The first column `scores[:, 0]` gives the scores conditioned on the predicted class.
                 The remaining columns `scores[:, i]` for `i = 1, . . ., m` gives the scores conditioned on `i - 1`
                 being the candidate true class for the test sample.
        """
        n_test = labels_pred_test.shape[0]
        # Query the index of `self.n_neighbors` nearest neighbors of each test sample
        if is_train:
            nn_indices, _ = self.index_knn.query_self(k=self.n_neighbors)
        else:
            nn_indices, _ = self.index_knn.query(features_test, k=self.n_neighbors)

        # Get the class label counts from the nearest neighbors of each sample
        _, data_counts = neighbors_label_counts(nn_indices, self.labels_train_enc, self.n_classes)

        scores = np.zeros(n_test, 1 + self.n_classes)
        preds_unique = self.labels_unique if (n_test > 1) else [labels_pred_test[0]]
        cnt_par = 0
        for c_hat in preds_unique:
            i = self.label_encoder([c_hat])[0]
            # Index of samples predicted into class `c_hat`
            ind = np.where(labels_pred_test == c_hat)[0]
            if ind.shape[0]:
                # Likelihood ratio statistic conditioned on the predicted class `c_hat`
                scores[ind, 0] = self.multinomial_lrt(data_counts[ind, :], self.proba_params_pred[i, :],
                                                      self.n_neighbors)
                cnt_par += ind.shape[0]
                if cnt_par >= n_test:
                    break

        for i, c in enumerate(self.labels_unique):
            # Likelihood ratio statistic conditioned on the candidate true class `c`
            scores[:, i + 1] = self.multinomial_lrt(data_counts, self.proba_params_true[i, :], self.n_neighbors)

        return scores

    @staticmethod
    def multinomial_lrt(data_counts, proba_params, n_neighbors):
        mat = data_counts * (np.log(np.clip(data_counts, 1e-16, None)) - np.log(n_neighbors * proba_params))
        return np.sum(mat, axis=1)


class LIDScore(TestStatistic):
    """
    Class that calculates the local intrinsic dimensionality (LID) based test score given the layer embeddings
    and the predicted class of samples.
    """
    def __init__(self, **kwargs):
        super(LIDScore, self).__init__(
            neighborhood_constant=kwargs.get('neighborhood_constant', NEIGHBORHOOD_CONST),
            n_neighbors=kwargs.get('n_neighbors', None),
            metric='euclidean',     # Has to be 'euclidean' for LID estimation
            shared_nearest_neighbors=False,     # Intentionally set to False
            approx_nearest_neighbors=kwargs.get('approx_nearest_neighbors', True),
            n_jobs=kwargs.get('n_jobs', 1),
            low_memory=kwargs.get('low_memory', False),
            seed_rng=kwargs.get('seed_rng', SEED_DEFAULT)
        )

        # Median LID estimate of the samples predicted into each class
        self.lid_median_pred = None
        # Median LID estimate of the samples labeled from each class
        self.lid_median_true = None
        # Scores for the training data
        self.scores_train = None

    def fit(self, features, labels, labels_pred, labels_unique=None):
        """
        Use the given feature vectors, true labels, and predicted labels to estimate the scoring model.

        :param features: numpy array of shape `(N, d)` where `N` and `d` are the number of samples and
                         dimension respectively.
        :param labels: numpy array of shape `(N, )` with the true labels per sample.
        :param labels_pred: numpy array of shape `(N, )` with the predicted labels per sample.
        :param labels_unique: None or a numpy array with the unique labels. For example, np.arange(1, 11). This can
                              be supplied as input during repeated calls to avoid having the find the unique
                              labels each time.
        :return:
            scores_train: numpy array of two scores for each training sample. Has shape `(N, 2)`, where the first
                          column gives the scores conditioned on the predicted class, and the second column gives
                          the scores conditioned on the true class.
        """
        super(LIDScore, self).fit(features, labels, labels_pred, labels_unique=labels_unique)

        # Build the KNN index for the given feature vectors
        self.index_knn = KNNIndex(
            features, n_neighbors=self.n_neighbors,
            metric=self.metric, metric_kwargs=self.metric_kwargs,
            shared_nearest_neighbors=self.shared_nearest_neighbors,
            approx_nearest_neighbors=self.approx_nearest_neighbors,
            n_jobs=self.n_jobs,
            low_memory=self.low_memory,
            seed_rng=self.seed_rng
        )
        # Indices and distances of the nearest neighbors of the points from `features`
        nn_indices, nn_distances = self.index_knn.query_self(k=self.n_neighbors)

        # Get the LID estimates for all samples based on their nearest neighbors distances
        lid_estimates = lid_mle_amsaleg(nn_distances)

        self.lid_median_pred = np.ones(self.n_classes)
        self.lid_median_true = np.ones(self.n_classes)
        # First score is conditioned the predicted class and the second score is conditioned on the true class
        self.scores_train = np.zeros(self.n_train, 2)
        for c in self.labels_unique:
            i = self.label_encoder([c])[0]
            # LID values of samples predicted into class `c`
            ind = np.where(labels_pred == c)[0]
            if ind.shape[0]:
                self.lid_median_pred[i] = np.median(lid_estimates[ind])
                self.scores_train[ind, 0] = lid_estimates[ind] / self.lid_median_pred[i]
            else:
                logger.warning("No samples are predicted into class '{}'. Setting the median LID "
                               "value to 1.".format(c))

            # LID values of samples labeled as class `c`
            ind = np.where(labels == c)[0]
            if ind.shape[0]:
                self.lid_median_true[i] = np.median(lid_estimates[ind])
                self.scores_train[ind, 1] = lid_estimates[ind] / self.lid_median_true[i]
            else:
                logger.warning("No samples are labeled from class '{}'. Setting the median LID "
                               "value to 1.".format(c))

        return self.scores_train

    def score(self, features_test, labels_pred_test, is_train=False):
        """
        Given the test feature vectors and their corresponding predicted labels, calculate a vector of scores for
        each test sample. Set `is_train = True` only if the `fit` method was called using `features_test`.

        :param features_test: numpy array of shape `(N, d)` where `N` and `d` are the number of samples and
                              dimension respectively.
        :param labels_pred_test: numpy array of shape `(N, )` with the predicted labels per sample.
        :param  is_train: Set to True if points from `features_test` were used for training, i.e. by the fit method.

        :return: numpy array of shape `(N, m + 1)` with a vector of scores for each sample, where `m` is the number
                 of classes. The first column `scores[:, 0]` gives the scores conditioned on the predicted class.
                 The remaining columns `scores[:, i]` for `i = 1, . . ., m` gives the scores conditioned on `i - 1`
                 being the candidate true class for the test sample.
        """
        n_test = labels_pred_test.shape[0]
        # Query the index of `self.n_neighbors` nearest neighbors of each test sample
        if is_train:
            nn_indices, nn_distances = self.index_knn.query_self(k=self.n_neighbors)
        else:
            nn_indices, nn_distances = self.index_knn.query(features_test, k=self.n_neighbors)

        # LID estimates of the test samples
        lid_estimates = lid_mle_amsaleg(nn_distances)

        scores = np.zeros(n_test, 1 + self.n_classes)
        preds_unique = self.labels_unique if (n_test > 1) else [labels_pred_test[0]]
        cnt_par = 0
        for c_hat in preds_unique:
            i = self.label_encoder([c_hat])[0]
            # Index of samples predicted into class `c_hat`
            ind = np.where(labels_pred_test == c_hat)[0]
            if ind.shape[0]:
                # LID scores normalized by the median LID value for the samples predicted into class `c`
                scores[ind, 0] = lid_estimates[ind] / self.lid_median_pred[i]

                cnt_par += ind.shape[0]
                if cnt_par >= n_test:
                    break

        for i, c in enumerate(self.labels_unique):
            # LID scores normalized by the median LID value for the samples labeled as class `c`
            scores[:, i + 1] = lid_estimates / self.lid_median_true[i]

        return scores
