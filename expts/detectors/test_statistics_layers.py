"""
Test statistics to be calculated at the different layers of the trained deep neural network.

"""
import numpy as np
import sys
from abc import ABC, abstractmethod
import multiprocessing
from functools import partial
from numba import njit, prange
from scipy.stats import binom
import logging
import copy
from sklearn.preprocessing import MinMaxScaler
from helpers.knn_index import KNNIndex, helper_knn_distance
from helpers.knn_classifier import neighbors_label_counts
from helpers.multinomial import (
    multinomial_estimation,
    special_dirichlet_prior
)
from helpers.lid_estimators import lid_mle_amsaleg
from helpers.dimension_reduction_methods import (
    pca_wrapper,
    helper_reconstruction_error
)
from detectors.pvalue_estimation import (
    pvalue_score,
    pvalue_score_all_pairs
)
from detectors.localized_pvalue_estimation import averaged_KLPE_anomaly_detection
from helpers.utils import get_num_jobs
from helpers.constants import (
    NEIGHBORHOOD_CONST,
    SEED_DEFAULT,
    METRIC_DEF,
    NUM_BOOTSTRAP,
    PCA_CUTOFF
)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
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
    def fit(self, features, labels, labels_pred, labels_unique=None, bootstrap=False):
        """
        Use the given feature vectors, true labels, and predicted labels to estimate the scoring model.

        :param features: numpy array of shape `(N, d)` where `N` and `d` are the number of samples and
                         dimension respectively.
        :param labels: numpy array of shape `(N, )` with the true labels per sample.
        :param labels_pred: numpy array of shape `(N, )` with the predicted labels per sample.
        :param labels_unique: None or a numpy array with the unique labels. For example, np.arange(1, 11). This can
                              be supplied as input during repeated calls to avoid having the find the unique
                              labels each time.
        :param bootstrap: Set to True in order to calculate a bootstrap resampled estimate of the p-value.
                          The default value is False because the p-values returned by the fit method are usually not
                          used down the line. Not using the bootstrap here makes it faster.
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
    def score(self, features_test, labels_pred_test, is_train=False, log_transform=True, bootstrap=True):
        """
        Given the feature vector and predicted labels for `N` samples, calculate their test statistic scores.

        :param features_test: numpy array of shape `(N, d)` where `N` and `d` are the number of samples and
                              dimension respectively.
        :param labels_pred_test: numpy array of shape `(N, )` with the predicted labels per sample.
        :param is_train: Set to True if points from `features_test` were used for training, i.e. by the fit method.
                         This is used to remove points from their own set of nearest neighbors.
        :param log_transform: Set to True to apply negative log transformation to the p-values.
        :param bootstrap: Set to True to calculate a bootstrap resampled estimate of the p-value.
        :return:
        """
        raise NotImplementedError


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
        # Index of train samples from each class based on the true class and predicted class
        self.indices_true = dict()
        self.indices_pred = dict()
        # Boolean mask indicating which subset of classes will be used for the test statistic.
        # This differs depending on the predicted class and the true class
        self.mask_included_pred = None
        self.mask_included_true = None
        # Type of test statistic (multinomial LRT or binomial deviation) to use given the predicted and true class
        self.type_test_stat_pred = None
        self.type_test_stat_true = None

    def fit(self, features, labels, labels_pred, labels_unique=None, bootstrap=False,
            n_classes_multinom=None, combine_low_proba_classes=False):
        """
        Use the given feature vectors, true labels, and predicted labels to estimate the scoring model.

        :param features: numpy array of shape `(N, d)` where `N` and `d` are the number of samples and
                         dimension respectively.
        :param labels: numpy array of shape `(N, )` with the true labels per sample.
        :param labels_pred: numpy array of shape `(N, )` with the predicted labels per sample.
        :param labels_unique: None or a numpy array with the unique labels. For example, np.arange(1, 11). This can
                              be supplied as input during repeated calls to avoid having the find the unique
                              labels each time.
        :param bootstrap: Set to True in order to calculate a bootstrap resampled estimate of the p-value.
                          The default value is False because the p-values returned by the fit method are usually not
                          used down the line. Not using the bootstrap here makes it faster.

        The optional arguments below allow you to combine low probability classes into one bigger class while
        calculating the multinomial likelihood ratio test statistic. This can be useful when the dataset has a
        large number of classes.
        Specify one of the options `n_classes_multinom` or `combine_low_proba_classes`:
        - The option `n_classes_multinom` allows you to explicitly specify the number of classes to include. If it
          is specified, then the option `combine_low_proba_classes` is ignored.
        - If the option `combine_low_proba_classes` is set to True, classes with very low probability are grouped
          into one bigger class. This can help reduce noise in the detection.
        :param n_classes_multinom: None or an int value >= 1. If `None`, then all classes will be distinct in the
                                   multinomial likelihood ratio test statistic. Otherwise, only the specified number
                                   of classes with highest probability will be kept distinct, and the rest of the
                                   classes will be grouped into one.
        :param combine_low_proba_classes: Set to True to automatically group low probability classes into one
                                          bigger class. This option will be used only if `n_classes_multinom = None`.

        :return: (scores, p_values)
            scores: numpy array of shape `(N, m + 1)` with a vector of scores for each sample, where `m` is the
                    number of classes. The first column `scores[:, 0]` gives the scores conditioned on the predicted
                    class. The remaining columns `scores[:, i]` for `i = 1, . . ., m` gives the scores conditioned
                    on `i - 1` being the candidate true class for the test sample.
            p_values: numpy array of same shape as `scores` containing the negative-log-transformed empirical
                      p-values of the scores.
        """
        super(MultinomialScore, self).fit(features, labels, labels_pred, labels_unique=labels_unique)

        # Number of classes to use for the multinomial likelihood ratio test statistic
        if n_classes_multinom is not None:
            combine_low_proba_classes = False
            n_classes_multinom = int(n_classes_multinom)
            if n_classes_multinom > self.n_classes:
                logger.warning("Argument 'n_classes_multinom' cannot be larger than the number of classes {:d}. "
                               "Setting it equal to the number of classes.".format(self.n_classes))
                n_classes_multinom = self.n_classes
            if n_classes_multinom < 1:
                raise ValueError("Invalid value {:d} for the argument 'n_classes_multinom'".format(n_classes_multinom))

        logger.info("Building a KNN index for nearest neighbor queries.")
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

        logger.info("Calculating the class label counts in the neighborhood of each sample and performing "
                    "multinomial parameter estimation.")
        self.labels_train_enc = self.label_encoder(labels)
        _, self.data_counts_train = neighbors_label_counts(nn_indices, self.labels_train_enc, self.n_classes)

        # Dirichlet prior counts for MAP estimation
        alpha_diric = special_dirichlet_prior(self.n_classes)

        # Probability parameters conditioned on the predicted class and the true class
        sz = (self.n_classes, self.n_classes)
        self.proba_params_pred = np.full(sz, 1. / self.n_classes)
        self.proba_params_true = np.full(sz, 1. / self.n_classes)
        # Class inclusion mask conditioned on the predicted and the true class
        self.mask_included_pred = np.full(sz, True)
        self.mask_included_true = np.full(sz, True)
        # Type of test statistic conditioned on the predicted and true class
        self.type_test_stat_pred = ['multi'] * self.n_classes
        self.type_test_stat_true = ['multi'] * self.n_classes
        for i, c in enumerate(self.labels_unique):
            # Index of samples predicted into class `c`
            ind = np.where(labels_pred == c)[0]
            self.indices_pred[c] = ind
            if ind.shape[0]:
                # Estimate the multinomial probability parameters given the predicted class `c`
                self.proba_params_pred[i, :] = multinomial_estimation(self.data_counts_train[ind, :],
                                                                      alpha_prior=alpha_diric[i, :])
            else:
                logger.warning("No samples are predicted into class '{}'. Skipping multinomial parameter "
                               "estimation and assigning uniform probabilities.".format(c))

            # Set the classes to include as distinct classes in test statistic given the predicted class `c`
            self.mask_included_pred[i, :], num_incl = self.set_distinct_classes(
                self.proba_params_pred[i, :], i, self.n_classes, n_classes_multinom, combine_low_proba_classes
            )
            if self.n_classes == 2 or num_incl == 1:
                # In the case of two distinct classes, we use a binomial test statistic
                self.type_test_stat_pred[i] = 'binom'

            if num_incl < self.n_classes:
                logger.info("Predicted class {}: {:d} distinct class(es). Grouping the remaining {:d} class(es)".
                            format(c, num_incl, self.n_classes - num_incl))

            # Index of samples with class label `c`
            ind = np.where(labels == c)[0]
            self.indices_true[c] = ind
            if ind.shape[0]:
                # Estimate the multinomial probability parameters given the true class `c`
                self.proba_params_true[i, :] = multinomial_estimation(self.data_counts_train[ind, :],
                                                                      alpha_prior=alpha_diric[i, :])
            else:
                # Unexpected, should not occur in practice
                logger.warning("No labeled samples from class '{}'. Skipping multinomial parameter estimation "
                               "and assigning uniform probabilities.".format(c))

            # Set the classes to include as distinct classes in the test statistic givene the true class `c`
            self.mask_included_true[i, :], num_incl = self.set_distinct_classes(
                self.proba_params_true[i, :], i, self.n_classes, n_classes_multinom, combine_low_proba_classes
            )
            if self.n_classes == 2 or num_incl == 1:
                # In the case of two distinct classes, we use a binomial test statistic
                self.type_test_stat_true[i] = 'binom'

            if num_incl < self.n_classes:
                logger.info("     True class {}: {:d} distinct class(es). Grouping the remaining {:d} class(es)".
                            format(c, num_incl, self.n_classes - num_incl))

        # Calculate the scores and p-values for each sample
        self.scores_train, p_values = self.score(features, labels_pred, is_train=True, bootstrap=bootstrap)
        return self.scores_train, p_values

    def score(self, features_test, labels_pred_test, is_train=False, log_transform=True, bootstrap=True):
        """
        Given the test feature vectors and their corresponding predicted labels, calculate a vector of scores for
        each test sample. Set `is_train = True` only if the `fit` method was called using `features_test`.

        :param features_test: numpy array of shape `(N, d)` where `N` and `d` are the number of samples and
                              dimension respectively.
        :param labels_pred_test: numpy array of shape `(N, )` with the predicted labels per sample.
        :param is_train: Set to True if points from `features_test` were used for training, i.e. by the fit method.
        :param log_transform: Set to True to apply negative log transformation to the p-values.
        :param bootstrap: Set to True to calculate a bootstrap resampled estimate of the p-value.

        :return: (scores, p_values)
            scores: numpy array of shape `(N, m + 1)` with a vector of scores for each sample, where `m` is the
                    number of classes. The first column `scores[:, 0]` gives the scores conditioned on the predicted
                    class. The remaining columns `scores[:, i]` for `i = 1, . . ., m` gives the scores conditioned
                    on `i - 1` being the candidate true class for the test sample.
            p_values: numpy array of same shape as `scores` containing the negative-log-transformed empirical
                      p-values of the scores.
        """
        n_test = labels_pred_test.shape[0]
        # Get the class label counts from the nearest neighbors of each sample
        if is_train:
            data_counts = self.data_counts_train
        else:
            nn_indices, _ = self.index_knn.query(features_test, k=self.n_neighbors)
            _, data_counts = neighbors_label_counts(nn_indices, self.labels_train_enc, self.n_classes)

        scores = np.zeros((n_test, 1 + self.n_classes))
        p_values = np.zeros((n_test, 1 + self.n_classes))
        preds_unique = self.labels_unique if (n_test > 1) else [labels_pred_test[0]]
        cnt_par = 0
        for c_hat in preds_unique:
            i = self.label_encoder([c_hat])[0]
            # Index of samples predicted into class `c_hat`
            ind = np.where(labels_pred_test == c_hat)[0]
            if ind.shape[0]:
                if self.type_test_stat_pred[i] == 'multi':
                    # Likelihood ratio statistic conditioned on the predicted class `c_hat`
                    scores[ind, 0] = self.multinomial_lrt(
                        data_counts[ind, :], self.proba_params_pred[i, :], self.mask_included_pred[i, :],
                        self.n_neighbors
                    )
                else:
                    # Score is the proportion of samples in the neighborhood that have a different label than the
                    # predicted class `c_hat`
                    scores[ind, 0] = (1. / self.n_neighbors) * (self.n_neighbors - data_counts[ind, i])

                # Empirical p-value estimates
                if not is_train:
                    p_values[ind, 0] = pvalue_score(
                        self.scores_train[self.indices_pred[c_hat], 0], scores[ind, 0], log_transform=log_transform,
                        bootstrap=bootstrap
                    )
                else:
                    p_values[ind, 0] = pvalue_score(
                        scores[ind, 0], scores[ind, 0], log_transform=log_transform, bootstrap=bootstrap
                    )

                cnt_par += ind.shape[0]
                if cnt_par >= n_test:
                    break

        for i, c in enumerate(self.labels_unique):
            if self.type_test_stat_true[i] == 'multi':
                # Likelihood ratio statistic conditioned on the candidate true class `c`
                scores[:, i + 1] = self.multinomial_lrt(
                    data_counts, self.proba_params_true[i, :], self.mask_included_true[i, :], self.n_neighbors
                )
            else:
                # Score is the proportion of samples in the neighborhood that have a different label from `c`
                scores[:, i + 1] = (1. / self.n_neighbors) * (self.n_neighbors - data_counts[:, i])

            # Empirical p-value estimates
            if not is_train:
                p_values[:, i + 1] = pvalue_score(
                    self.scores_train[self.indices_true[c], i + 1], scores[:, i + 1], log_transform=log_transform,
                    bootstrap=bootstrap
                )
            else:
                p_values[:, i + 1] = pvalue_score(
                    scores[self.indices_true[c], i + 1], scores[:, i + 1], log_transform=log_transform,
                    bootstrap=bootstrap
                )

        return scores, p_values

    @staticmethod
    def multinomial_lrt(data_counts, proba_params, mask_classes, n_neighbors):
        # Multinomial likelihood ratio test statistic
        if np.all(mask_classes):
            # All classes are distinct
            mat = data_counts * (np.log(np.clip(data_counts, sys.float_info.epsilon, None)) -
                                 np.log(n_neighbors * proba_params))
        else:
            # Counts from the distinct classes
            data_counts_sel = data_counts[:, mask_classes]
            # Adjoin the cumulative count from the remaining classes as a new column
            counts_rem = n_neighbors - np.sum(data_counts_sel, axis=1)
            data_counts_sel = np.hstack((data_counts_sel, counts_rem[:, np.newaxis]))

            # Probability parameters from the distinct classes
            p = proba_params[mask_classes]
            # Append the cumulative probability from the remaining classes
            p = np.append(p, max(1. - np.sum(p), sys.float_info.epsilon))

            mat = data_counts_sel * (np.log(np.clip(data_counts_sel, sys.float_info.epsilon, None)) -
                                     np.log(n_neighbors * p))

        return np.sum(mat, axis=1)

    @staticmethod
    def set_distinct_classes(proba, ind_class, n_classes, n_classes_multinom, combine_low_proba_classes):
        if n_classes_multinom is None:
            if not combine_low_proba_classes:
                # All classes are kept distinct in the default setting
                mask_incl = np.ones(n_classes, dtype=np.bool)
            else:
                mask_incl = np.zeros(n_classes, dtype=np.bool)
                # Sort the classes by increasing probability and combine the classes that have a cumulative
                # probability below `1 / n_classes` into one group
                tmp_ind = np.argsort(proba)
                v = np.cumsum(proba[tmp_ind])
                tmp_ind = tmp_ind[v >= (1. / n_classes)]
                mask_incl[tmp_ind] = True
                # class corresponding to `ind_class` is always kept distinct
                mask_incl[ind_class] = True
        else:
            mask_incl = np.zeros(n_classes, dtype=np.bool)
            # class corresponding to `ind_class` is always kept distinct
            mask_incl[ind_class] = True
            if n_classes_multinom > 1:
                # Sort the classes in decreasing order of probability
                v = copy.copy(proba)
                v[ind_class] = -1.
                tmp_ind = np.argsort(v)[::-1]
                # Include the required number of classes with highest probability
                mask_incl[tmp_ind[:(n_classes_multinom - 1)]] = True

        return mask_incl, np.sum(mask_incl)


class BinomialScore(TestStatistic):
    """
    Class that calculates the binomial test statistic from the class counts observed in the `k` nearest neighbors
    of a sample.
    """
    def __init__(self, **kwargs):
        super(BinomialScore, self).__init__(
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
        # Binomial probability parameter conditioned on the predicted class
        self.proba_params_pred = None
        # Binomial probability parameter conditioned on the true class
        self.proba_params_true = None
        # Likelihood ratio statistic scores for the training data
        self.scores_train = None
        # Index of train samples from each class based on the true class and predicted class
        self.indices_true = dict()
        self.indices_pred = dict()

    def fit(self, features, labels, labels_pred, labels_unique=None, bootstrap=False):
        """
        Use the given feature vectors, true labels, and predicted labels to estimate the scoring model.
        :param features: numpy array of shape `(N, d)` where `N` and `d` are the number of samples and
                         dimension respectively.
        :param labels: numpy array of shape `(N, )` with the true labels per sample.
        :param labels_pred: numpy array of shape `(N, )` with the predicted labels per sample.
        :param labels_unique: None or a numpy array with the unique labels. For example, np.arange(1, 11). This can
                              be supplied as input during repeated calls to avoid having the find the unique
                              labels each time.
        :param bootstrap: Set to True in order to calculate a bootstrap resampled estimate of the p-value.
                          The default value is False because the p-values returned by the fit method are usually not
                          used down the line. Not using the bootstrap here makes it faster.

        :return: (scores, p_values)
            scores: numpy array of shape `(N, m + 1)` with a vector of scores for each sample, where `m` is the
                    number of classes. The first column `scores[:, 0]` gives the scores conditioned on the predicted
                    class. The remaining columns `scores[:, i]` for `i = 1, . . ., m` gives the scores conditioned
                    on `i - 1` being the candidate true class for the test sample.
            p_values: numpy array of same shape as `scores` containing the negative-log-transformed empirical
                      p-values of the scores.
        """
        super(BinomialScore, self).fit(features, labels, labels_pred, labels_unique=labels_unique)

        logger.info("Building a KNN index for nearest neighbor queries.")
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

        logger.info("Calculating the class label counts in the neighborhood of each sample and performing "
                    "binomial parameter estimation.")
        self.labels_train_enc = self.label_encoder(labels)
        _, self.data_counts_train = neighbors_label_counts(nn_indices, self.labels_train_enc, self.n_classes)

        # Parameter estimation conditioned on the predicted class and the true class
        self.proba_params_pred = 0.5 * np.ones(self.n_classes)
        self.proba_params_true = 0.5 * np.ones(self.n_classes)
        for i, c in enumerate(self.labels_unique):
            # Index of samples predicted into class `c`
            ind = np.where(labels_pred == c)[0]
            self.indices_pred[c] = ind
            if ind.shape[0]:
                # Estimate the binomial probability parameter given the predicted class `c`
                counts_pos = self.data_counts_train[ind, i]
                self.proba_params_pred[i] = self.binomial_estimation(counts_pos, self.n_neighbors)
            else:
                logger.warning("No samples are predicted into class '{}'. Skipping binomial parameter estimation "
                               "and setting the probability parameter to 0.5.".format(c))

            # Index of samples with class label `c`
            ind = np.where(labels == c)[0]
            self.indices_true[c] = ind
            if ind.shape[0]:
                # Estimate the binomial probability parameter given the true class `c`
                counts_pos = self.data_counts_train[ind, i]
                self.proba_params_true[i] = self.binomial_estimation(counts_pos, self.n_neighbors)
            else:
                # Unexpected, should not occur in practice
                logger.warning("No labeled samples from class '{}'. Skipping binomial parameter estimation "
                               "and setting the probability parameter to 0.5.".format(c))

        # Calculate the scores and p-values for each sample
        self.scores_train, p_values = self.score(features, labels_pred, is_train=True, bootstrap=bootstrap)
        return self.scores_train, p_values

    def score(self, features_test, labels_pred_test, is_train=False, log_transform=True, bootstrap=True):
        """
        Given the test feature vectors and their corresponding predicted labels, calculate a vector of scores for
        each test sample. Set `is_train = True` only if the `fit` method was called using `features_test`.

        :param features_test: numpy array of shape `(N, d)` where `N` and `d` are the number of samples and
                              dimension respectively.
        :param labels_pred_test: numpy array of shape `(N, )` with the predicted labels per sample.
        :param is_train: Set to True if points from `features_test` were used for training, i.e. by the fit method.
        :param log_transform: Set to True to apply negative log transformation to the p-values.
        :param bootstrap: Set to True to calculate a bootstrap resampled estimate of the p-value.

        :return: (scores, p_values)
            scores: numpy array of shape `(N, m + 1)` with a vector of scores for each sample, where `m` is the
                    number of classes. The first column `scores[:, 0]` gives the scores conditioned on the predicted
                    class. The remaining columns `scores[:, i]` for `i = 1, . . ., m` gives the scores conditioned
                    on `i - 1` being the candidate true class for the test sample.
            p_values: numpy array of same shape as `scores` containing the negative-log-transformed empirical
                      p-values of the scores.
        """
        n_test = labels_pred_test.shape[0]
        # Get the class label counts from the nearest neighbors of each sample
        if is_train:
            data_counts = self.data_counts_train
        else:
            nn_indices, _ = self.index_knn.query(features_test, k=self.n_neighbors)
            _, data_counts = neighbors_label_counts(nn_indices, self.labels_train_enc, self.n_classes)

        scores = np.zeros((n_test, 1 + self.n_classes))
        p_values = np.zeros((n_test, 1 + self.n_classes))
        preds_unique = self.labels_unique if (n_test > 1) else [labels_pred_test[0]]
        cnt_par = 0
        for c_hat in preds_unique:
            i = self.label_encoder([c_hat])[0]
            # Index of samples predicted into class `c_hat`
            ind = np.where(labels_pred_test == c_hat)[0]
            if ind.shape[0]:
                # Score is the proportion of samples in the neighborhood that have a different label than the
                # predicted class `c_hat`
                scores[ind, 0] = (1. / self.n_neighbors) * (self.n_neighbors - data_counts[ind, i])

                # Empirical p-value estimates
                if not is_train:
                    # `self.scores_train[self.indices_pred[c_hat], 0]` are the scores from the training data that
                    # were predicted into class `c_hat`
                    p_values[ind, 0] = pvalue_score(
                        self.scores_train[self.indices_pred[c_hat], 0], scores[ind, 0], log_transform=log_transform,
                        bootstrap=bootstrap
                    )
                else:
                    p_values[ind, 0] = pvalue_score(
                        scores[ind, 0], scores[ind, 0], log_transform=log_transform, bootstrap=bootstrap
                    )

                cnt_par += ind.shape[0]
                if cnt_par >= n_test:
                    break

        for i, c in enumerate(self.labels_unique):
            # Score is the proportion of samples in the neighborhood that have a different label from `c`
            scores[:, i + 1] = (1. / self.n_neighbors) * (self.n_neighbors - data_counts[:, i])

            # Empirical p-value estimates
            if not is_train:
                p_values[:, i + 1] = pvalue_score(
                    self.scores_train[self.indices_true[c], i + 1], scores[:, i + 1], log_transform=log_transform,
                    bootstrap=bootstrap
                )
            else:
                p_values[:, i + 1] = pvalue_score(
                    scores[self.indices_true[c], i + 1], scores[:, i + 1], log_transform=log_transform,
                    bootstrap=bootstrap
                )

        return scores, p_values

    @staticmethod
    def binomial_estimation(counts_pos, n_trials, a=1.001, b=1.001):
        """
        Maximum-a-posteriori (MAP) estimation for the binomial distribution.
        `a` and `b` are constants of a beta distribution which is a conjugate prior for the binomial distribution.
        Setting `a = b = 1` corresponds to maximum likelihood estimation. Larger values of `a` and `b` would act like
        pseudo counts for the two categories of the binomial distribution.

        :param counts_pos: numpy array with the counts corresponding to the positive outcome. Should have shape `(n, )`,
                           where `n` is the number of samples.
        :param n_trials: int value >= 1 which is the number of binomial trials. This should be >= `np.max(counts_pos)`
                         for obvious reasons.
        :param a: value >= 1 that is a parameter of the beta prior.
        :param b: value >= 1 that is a parameter of the beta prior.

        :return: estimated binomial probability parameter.
        """
        if np.max(counts_pos) > n_trials:
            raise ValueError("Positive count value cannot be larger than the maximum value {:d}".format(n_trials))

        return (a - 1. + np.sum(counts_pos)) / (a + b - 2. + n_trials * counts_pos.shape[0])


class LIDScore(TestStatistic):
    """
    Class that calculates the local intrinsic dimensionality (LID) based test score given the layer embeddings
    and the predicted class of samples.

    NOTE: For this method, do not apply dimension reduction to the layer embeddings. We want to estimate the LID
    values in the original feature space.
    """
    def __init__(self, **kwargs):
        super(LIDScore, self).__init__(
            neighborhood_constant=kwargs.get('neighborhood_constant', NEIGHBORHOOD_CONST),
            n_neighbors=kwargs.get('n_neighbors', None),
            metric=kwargs.get('metric', 'euclidean'),       # use 'euclidean' metric for LID estimation
            metric_kwargs=kwargs.get('metric_kwargs', None),
            shared_nearest_neighbors=False,     # Intentionally set to False
            approx_nearest_neighbors=kwargs.get('approx_nearest_neighbors', True),
            n_jobs=kwargs.get('n_jobs', 1),
            low_memory=kwargs.get('low_memory', False),
            seed_rng=kwargs.get('seed_rng', SEED_DEFAULT)
        )

        # Feature means and PCA transformation matrix
        self.mean_data = None
        self.transform_pca = None
        # Number of neighbors can be set based on the number of samples per class if `self.n_neighbors = None`
        self.n_neighbors_per_class = dict()
        # KNN indices for the data points from each class
        self.index_knn = dict()
        # LID estimates conditioned on different predicted and true classes for the train samples
        self.lid_estimates_train = None
        # Scores for the training data
        self.scores_train = None
        # Index of train samples from each class based on the true class and predicted class
        self.indices_true = dict()
        self.indices_pred = dict()

    def fit(self, features, labels, labels_pred, labels_unique=None, bootstrap=False,
            min_dim_pca=10000, pca_cutoff=PCA_CUTOFF):
        """
        Use the given feature vectors, true labels, and predicted labels to estimate the scoring model.

        :param features: numpy array of shape `(N, d)` where `N` and `d` are the number of samples and
                         dimension respectively.
        :param labels: numpy array of shape `(N, )` with the true labels per sample.
        :param labels_pred: numpy array of shape `(N, )` with the predicted labels per sample.
        :param labels_unique: None or a numpy array with the unique labels. For example, np.arange(1, 11). This can
                              be supplied as input during repeated calls to avoid having the find the unique
                              labels each time.
        :param bootstrap: Set to True in order to calculate a bootstrap resampled estimate of the p-value.
                          The default value is False because the p-values returned by the fit method are usually not
                          used down the line. Not using the bootstrap here makes it faster.
        :param min_dim_pca: (int) Minimum dimension above which PCA pre-processing is applied.
        :param pca_cutoff: Float value in (0, 1] specifying the proportion of cumulative data variance to preserve
                           in the projected (dimension-reduced) data.

        :return: (scores, p_values)
            scores: numpy array of shape `(N, m + 1)` with a vector of scores for each sample, where `m` is the
                    number of classes. The first column `scores[:, 0]` gives the scores conditioned on the predicted
                    class. The remaining columns `scores[:, i]` for `i = 1, . . ., m` gives the scores conditioned
                    on `i - 1` being the candidate true class for the test sample.
            p_values: numpy array of same shape as `scores` containing the negative-log-transformed empirical
                      p-values of the scores.
        """
        set_n_neighbors = True if (self.n_neighbors is None) else False
        # `fit` method of the super class
        super(LIDScore, self).fit(features, labels, labels_pred, labels_unique=labels_unique)

        if self.dim >= min_dim_pca:
            logger.info("Data dimension = {:d}. Applying PCA as a pre-processing step.".format(self.dim))
            features, self.mean_data, self.transform_pca = pca_wrapper(
                features, cutoff=pca_cutoff, seed_rng=self.seed_rng
            )
            self.dim = features.shape[1]

        # Column 0 corresponds to the LID estimates conditioned on the predicted class.
        # Column `i` for `i = 1, 2, . . .` correspond to the LID estimates conditioned on the true class being `i - 1`
        self.lid_estimates_train = np.zeros((self.n_train, 1 + self.n_classes))

        logger.info("Building KNN indices for nearest neighbor queries from each class.")
        for i, c in enumerate(self.labels_unique):
            logger.info("Processing class {}:".format(c))
            # Labeled samples from class `c`
            ind = np.where(labels == c)[0]
            self.indices_true[c] = ind
            n_true = ind.shape[0]
            # Number of neighbors for the labeled samples from class `c`
            if set_n_neighbors:
                self.n_neighbors_per_class[c] = int(np.ceil(n_true ** self.neighborhood_constant))
            else:
                self.n_neighbors_per_class[c] = self.n_neighbors

            if n_true:
                # KNN index for the labeled samples from class `c`
                self.index_knn[c] = KNNIndex(
                    features[ind, :], n_neighbors=self.n_neighbors_per_class[c],
                    metric=self.metric, metric_kwargs=self.metric_kwargs,
                    shared_nearest_neighbors=self.shared_nearest_neighbors,
                    approx_nearest_neighbors=self.approx_nearest_neighbors,
                    n_jobs=self.n_jobs,
                    low_memory=self.low_memory,
                    seed_rng=self.seed_rng
                )
                # LID estimates for the labeled samples from class `c`
                _, nn_distances = self.index_knn[c].query_self(k=self.n_neighbors_per_class[c])
                self.lid_estimates_train[ind, i + 1] = lid_mle_amsaleg(nn_distances)

                # Commenting this block out to avoid unneccessary computation.
                # The reconstruction errors for these samples are not used anywhere else.
                '''
                # LID estimates for the samples not labeled as class `c`
                ind_comp = np.where(labels != c)[0]
                _, nn_distances = self.index_knn[c].query(features[ind_comp, :], k=self.n_neighbors_per_class[c])
                self.lid_estimates_train[ind_comp, i + 1] = lid_mle_amsaleg(nn_distances)
                '''
            else:
                raise ValueError("No labeled samples from class '{}'. Cannot proceed.".format(c))

            # Samples predicted into class `c`
            ind = np.where(labels_pred == c)[0]
            self.indices_pred[c] = ind
            n_pred = ind.shape[0]
            if n_pred:
                # Distance to nearest neighbors of samples predicted into class `c` that are also labeled as
                # class `c`. These samples will be a part of the KNN index
                nn_distances = helper_knn_distance(self.indices_pred[c], self.indices_true[c], nn_distances)

                # Distance to nearest neighbors of samples predicted into class `c` that are not labeled as
                # class `c`. These samples will not be a part of the KNN index
                mask = (nn_distances[:, 0] < 0.)
                if np.any(mask):
                    ind_comp = self.indices_pred[c][mask]
                    _, temp_arr = self.index_knn[c].query(features[ind_comp, :], k=self.n_neighbors_per_class[c])
                    nn_distances[mask, :] = temp_arr

                # LID estimates from the k nearest neighbor distances of each sample
                self.lid_estimates_train[ind, 0] = lid_mle_amsaleg(nn_distances)
            else:
                raise ValueError("No predicted samples from class '{}'. Cannot proceed.".format(c))

        # Calculate the scores and p-values for each sample
        self.scores_train, p_values = self.score(features, labels_pred, is_train=True, bootstrap=bootstrap)
        return self.scores_train, p_values

    def score(self, features_test, labels_pred_test, is_train=False, log_transform=True, bootstrap=True):
        """
        Given the test feature vectors and their corresponding predicted labels, calculate a vector of scores for
        each test sample. Set `is_train = True` only if the `fit` method was called using `features_test`.

        :param features_test: numpy array of shape `(N, d)` where `N` and `d` are the number of samples and
                              dimension respectively.
        :param labels_pred_test: numpy array of shape `(N, )` with the predicted labels per sample.
        :param is_train: Set to True if points from `features_test` were used for training, i.e. by the fit method.
        :param log_transform: Set to True to apply negative log transformation to the p-values.
        :param bootstrap: Set to True to calculate a bootstrap resampled estimate of the p-value.

        :return: (scores, p_values)
            scores: numpy array of shape `(N, m + 1)` with a vector of scores for each sample, where `m` is the
                    number of classes. The first column `scores[:, 0]` gives the scores conditioned on the predicted
                    class. The remaining columns `scores[:, i]` for `i = 1, . . ., m` gives the scores conditioned
                    on `i - 1` being the candidate true class for the test sample.
            p_values: numpy array of same shape as `scores` containing the negative-log-transformed empirical
                      p-values of the scores.
        """
        n_test = labels_pred_test.shape[0]
        if (self.transform_pca is not None) and (not is_train):
            # Apply a PCA transformation to the test features
            features_test = np.dot(features_test - self.mean_data, self.transform_pca)

        scores = np.zeros((n_test, 1 + self.n_classes))
        p_values = np.zeros((n_test, 1 + self.n_classes))
        preds_unique = self.labels_unique if (n_test > 1) else [labels_pred_test[0]]
        cnt_par = 0
        for c_hat in preds_unique:
            # Index of samples predicted into class `c_hat`
            ind = np.where(labels_pred_test == c_hat)[0]
            if ind.shape[0]:
                if not is_train:
                    # LID estimate relative to the samples predicted into class `c_hat`
                    _, nn_distances = self.index_knn[c_hat].query(features_test[ind, :],
                                                                  k=self.n_neighbors_per_class[c_hat])
                    scores[ind, 0] = lid_mle_amsaleg(nn_distances)
                    # p-value of the LID estimates
                    p_values[ind, 0] = pvalue_score(
                        self.scores_train[self.indices_pred[c_hat], 0], scores[ind, 0], log_transform=log_transform,
                        bootstrap=bootstrap
                    )
                else:
                    # Precomputed LID estimates
                    scores[ind, 0] = self.lid_estimates_train[ind, 0]
                    # p-value of the LID estimates
                    p_values[ind, 0] = pvalue_score(
                        scores[ind, 0], scores[ind, 0], log_transform=log_transform, bootstrap=bootstrap
                    )

                cnt_par += ind.shape[0]
                if cnt_par >= n_test:
                    break

        for i, c in enumerate(self.labels_unique):
            if not is_train:
                # LID estimates relative to the samples labeled as class `c`
                _, nn_distances = self.index_knn[c].query(features_test, k=self.n_neighbors_per_class[c])
                scores[:, i + 1] = lid_mle_amsaleg(nn_distances)
                # p-value of the LID estimates
                p_values[:, i + 1] = pvalue_score(
                    self.scores_train[self.indices_true[c], i + 1], scores[:, i + 1], log_transform=log_transform,
                    bootstrap=bootstrap
                )
            else:
                # Precomputed LID estimates
                scores[:, i + 1] = self.lid_estimates_train[:, i + 1]
                # p-value of the LID estimates
                p_values[:, i + 1] = pvalue_score(
                    scores[self.indices_true[c], i + 1], scores[:, i + 1], log_transform=log_transform,
                    bootstrap=bootstrap
                )

        return scores, p_values


class DistanceScore(TestStatistic):
    """
    Average distance statistic from the k/2 to 3k/2 nearest neighbors of each point. This statistic is used in the
    localized p-value estimation paper.
    Qian, Jing, and Venkatesh Saligrama. "New statistic in p-value estimation for anomaly detection."
    IEEE Statistical Signal Processing Workshop (SSP). IEEE, 2012.
    """
    def __init__(self, **kwargs):
        super(DistanceScore, self).__init__(
            neighborhood_constant=kwargs.get('neighborhood_constant', NEIGHBORHOOD_CONST),
            n_neighbors=kwargs.get('n_neighbors', None),
            metric=kwargs.get('metric', METRIC_DEF),
            metric_kwargs=kwargs.get('metric_kwargs', None),
            shared_nearest_neighbors=False,     # Intentionally set to False
            approx_nearest_neighbors=kwargs.get('approx_nearest_neighbors', True),
            n_jobs=kwargs.get('n_jobs', 1),
            low_memory=kwargs.get('low_memory', False),
            seed_rng=kwargs.get('seed_rng', SEED_DEFAULT)
        )

        # Feature scaler
        self.scaler = None
        # Localize p-value estimation models for the set of samples from each predicted and true class
        self.klpe_models_pred = dict()
        self.klpe_models_true = dict()
        # Average distance score for the train samples
        self.distances_avg_train = None
        # Scores on the training data
        self.scores_train = None
        # Index of train samples from each class based on the true class and predicted class
        self.indices_true = dict()
        self.indices_pred = dict()

    def fit(self, features, labels, labels_pred, labels_unique=None, bootstrap=False):
        """
        Use the given feature vectors, true labels, and predicted labels to estimate the scoring model.

        :param features: numpy array of shape `(N, d)` where `N` and `d` are the number of samples and
                         dimension respectively.
        :param labels: numpy array of shape `(N, )` with the true labels per sample.
        :param labels_pred: numpy array of shape `(N, )` with the predicted labels per sample.
        :param labels_unique: None or a numpy array with the unique labels. For example, np.arange(1, 11). This can
                              be supplied as input during repeated calls to avoid having the find the unique
                              labels each time.
        :param bootstrap: Set to True in order to calculate a bootstrap resampled estimate of the p-value.
                          The default value is False because the p-values returned by the fit method are usually not
                          used down the line. Not using the bootstrap here makes it faster.
        :return: (scores, p_values)
            scores: numpy array of shape `(N, m + 1)` with a vector of scores for each sample, where `m` is the
                    number of classes. The first column `scores[:, 0]` gives the scores conditioned on the predicted
                    class. The remaining columns `scores[:, i]` for `i = 1, . . ., m` gives the scores conditioned
                    on `i - 1` being the candidate true class for the test sample.
            p_values: numpy array of same shape as `scores` containing the negative-log-transformed empirical
                      p-values of the scores.
        """
        set_n_neighbors = True if (self.n_neighbors is None) else False
        # `fit` method of the super class
        super(DistanceScore, self).fit(features, labels, labels_pred, labels_unique=labels_unique)

        # Scale the individual feature values to the range [-1, 1]
        # self.scaler = MinMaxScaler(feature_range=(-1, 1)).fit(features)
        # features = self.scaler.transform(features)

        # Column 0 corresponds to the average distance conditioned on the predicted class.
        # Column `i` for `i = 1, 2, . . .` correspond to the average distance conditioned on the true class
        # being `i - 1`
        self.distances_avg_train = np.zeros((self.n_train, 1 + self.n_classes))
        kwargs_lpe = {
            'standardize': False,
            'metric': self.metric,
            'metric_kwargs': self.metric_kwargs,
            'approx_nearest_neighbors': self.approx_nearest_neighbors,
            'n_jobs': self.n_jobs,
            'seed_rng': self.seed_rng
        }
        if set_n_neighbors:
            # number of neighbors is set based on the number of samples per class
            kwargs_lpe['neighborhood_constant'] = self.neighborhood_constant
        else:
            kwargs_lpe['n_neighbors'] = self.n_neighbors

        for i, c in enumerate(self.labels_unique):
            logger.info("Fitting localized p-value estimation models for class {}:".format(c))
            # Samples predicted into class `c`
            ind = np.where(labels_pred == c)[0]
            self.indices_pred[c] = ind
            if ind.shape[0]:
                self.klpe_models_pred[c] = averaged_KLPE_anomaly_detection(**kwargs_lpe)
                self.klpe_models_pred[c].fit(features[ind, :])
                self.distances_avg_train[ind, 0] = self.klpe_models_pred[c].dist_stat_nominal
            else:
                raise ValueError("No predicted samples from class '{}'. Cannot proceed.".format(c))

            # Labeled samples from class `c`
            ind = np.where(labels == c)[0]
            self.indices_true[c] = ind
            if ind.shape[0]:
                self.klpe_models_true[c] = averaged_KLPE_anomaly_detection(**kwargs_lpe)
                self.klpe_models_true[c].fit(features[ind, :])
                self.distances_avg_train[ind, i + 1] = self.klpe_models_true[c].dist_stat_nominal

                # Commenting this block out to avoid unneccessary computation.
                # The average distance for these samples is not used anywhere else.
                '''
                # Average distance values for samples not labeled as class `c`
                ind_comp = np.where(labels != c)[0]
                self.distances_avg_train[ind_comp, i + 1] = self.klpe_models_true[c].distance_statistic(
                    features[ind_comp, :]
                )
                '''
            else:
                raise ValueError("No labeled samples from class '{}'. Cannot proceed.".format(c))

        # Calculate the scores and p-values for each sample
        self.scores_train, p_values = self.score(features, labels_pred, is_train=True, bootstrap=bootstrap)
        return self.scores_train, p_values

    def score(self, features_test, labels_pred_test, is_train=False, log_transform=True, bootstrap=True):
        """
        Given the test feature vectors and their corresponding predicted labels, calculate a vector of scores for
        each test sample. Set `is_train = True` only if the `fit` method was called using `features_test`.

        :param features_test: numpy array of shape `(N, d)` where `N` and `d` are the number of samples and
                              dimension respectively.
        :param labels_pred_test: numpy array of shape `(N, )` with the predicted labels per sample.
        :param is_train: Set to True if points from `features_test` were used for training, i.e. by the fit method.
        :param log_transform: Set to True to apply negative log transformation to the p-values.
        :param bootstrap: Set to True to calculate a bootstrap resampled estimate of the p-value.

        :return: (scores, p_values)
            scores: numpy array of shape `(N, m + 1)` with a vector of scores for each sample, where `m` is the
                    number of classes. The first column `scores[:, 0]` gives the scores conditioned on the predicted
                    class. The remaining columns `scores[:, i]` for `i = 1, . . ., m` gives the scores conditioned
                    on `i - 1` being the candidate true class for the test sample.
            p_values: numpy array of same shape as `scores` containing the negative-log-transformed empirical
                      p-values of the scores.
        """
        n_test = labels_pred_test.shape[0]
        if not is_train:
            if self.scaler is not None:
                features_test = self.scaler.transform(features_test)

        scores = np.zeros((n_test, 1 + self.n_classes))
        p_values = np.zeros((n_test, 1 + self.n_classes))
        preds_unique = self.labels_unique if (n_test > 1) else [labels_pred_test[0]]
        cnt_par = 0
        for c_hat in preds_unique:
            # Index of samples predicted into class `c_hat`
            ind = np.where(labels_pred_test == c_hat)[0]
            if ind.shape[0]:
                if not is_train:
                    # Average distances and corresponding negative log p-values
                    p_values[ind, 0], scores[ind, 0] = self.klpe_models_pred[c_hat].score(features_test[ind, :],
                                                                                          return_distances=True)
                else:
                    # Precomputed average distances
                    scores[ind, 0] = self.distances_avg_train[ind, 0]
                    # Negative log p-values
                    p_values[ind, 0] = pvalue_score(
                        scores[ind, 0], scores[ind, 0], log_transform=log_transform, bootstrap=bootstrap
                    )

                cnt_par += ind.shape[0]
                if cnt_par >= n_test:
                    break

        for i, c in enumerate(self.labels_unique):
            if not is_train:
                # Average distances and corresponding negative log p-values
                p_values[:, i + 1], scores[:, i + 1] = self.klpe_models_true[c].score(features_test,
                                                                                      return_distances=True)
            else:
                # Precomputed average distances
                scores[:, i + 1] = self.distances_avg_train[:, i + 1]
                # Negative log p-values
                p_values[:, i + 1] = pvalue_score(
                    scores[self.indices_true[c], i + 1], scores[:, i + 1], log_transform=log_transform,
                    bootstrap=bootstrap
                )

        return scores, p_values


class TrustScore(TestStatistic):
    """
    Test statistic based on the trust score at each layer of the DNN.
    """
    def __init__(self, **kwargs):
        super(TrustScore, self).__init__(
            neighborhood_constant=kwargs.get('neighborhood_constant', NEIGHBORHOOD_CONST),
            n_neighbors=kwargs.get('n_neighbors', None),
            metric=kwargs.get('metric', METRIC_DEF),
            metric_kwargs=kwargs.get('metric_kwargs', None),
            shared_nearest_neighbors=False,     # Intentionally set to False
            approx_nearest_neighbors=kwargs.get('approx_nearest_neighbors', True),
            n_jobs=kwargs.get('n_jobs', 1),
            low_memory=kwargs.get('low_memory', False),
            seed_rng=kwargs.get('seed_rng', SEED_DEFAULT)
        )

        # Feature scaler
        self.scaler = None
        # Number of neighbors is set based on the number of samples per class if `self.n_neighbors = None`
        self.n_neighbors_pred = dict()
        self.n_neighbors_true = dict()
        # KNN indices for the data points from each predicted class and true class
        self.index_knn_pred = dict()
        self.index_knn_true = dict()
        # Distance ratio similar to the trust score calculated on the training data
        self.distance_ratio_train = None
        # Scores for the training data
        self.scores_train = None
        # Index of train samples from each class based on the true class and predicted class
        self.indices_true = dict()
        self.indices_pred = dict()

    def fit(self, features, labels, labels_pred, labels_unique=None, bootstrap=False):
        """
        Use the given feature vectors, true labels, and predicted labels to estimate the scoring model.
        :param features: numpy array of shape `(N, d)` where `N` and `d` are the number of samples and
                         dimension respectively.
        :param labels: numpy array of shape `(N, )` with the true labels per sample.
        :param labels_pred: numpy array of shape `(N, )` with the predicted labels per sample.
        :param labels_unique: None or a numpy array with the unique labels. For example, np.arange(1, 11). This can
                              be supplied as input during repeated calls to avoid having the find the unique
                              labels each time.
        :param bootstrap: Set to True in order to calculate a bootstrap resampled estimate of the p-value.
                          The default value is False because the p-values returned by the fit method are usually not
                          used down the line. Not using the bootstrap her makes it faster.

        :return: (scores, p_values)
            scores: numpy array of shape `(N, m + 1)` with a vector of scores for each sample, where `m` is the
                    number of classes. The first column `scores[:, 0]` gives the scores conditioned on the predicted
                    class. The remaining columns `scores[:, i]` for `i = 1, . . ., m` gives the scores conditioned
                    on `i - 1` being the candidate true class for the test sample.
            p_values: numpy array of same shape as `scores` containing the negative-log-transformed empirical
                      p-values of the scores.
        """
        set_n_neighbors = True if (self.n_neighbors is None) else False
        # `fit` method of the super class
        super(TrustScore, self).fit(features, labels, labels_pred, labels_unique=labels_unique)

        # Scale the individual feature values to the range [-1, 1]
        # self.scaler = MinMaxScaler(feature_range=(-1, 1)).fit(features)
        # features = self.scaler.transform(features)

        # Column 0 corresponds to the distance ratio conditioned on the predicted class.
        # Column `i` for `i = 1, 2, . . .` corresponds to the distance ratio conditioned on the true class
        # being `i - 1`
        self.distance_ratio_train = np.zeros((self.n_train, 1 + self.n_classes))

        logger.info("Building KNN indices for nearest neighbor queries from each predicted and each true class.")
        for i, c in enumerate(self.labels_unique):
            logger.info("Processing class {}:".format(c))
            # Samples predicted into class `c`
            ind = np.where(labels_pred == c)[0]
            self.indices_pred[c] = ind
            n_pred = ind.shape[0]
            # Number of neighbors for the samples predicted into class `c`
            if set_n_neighbors:
                self.n_neighbors_pred[c] = int(np.ceil(n_pred ** self.neighborhood_constant))
            else:
                self.n_neighbors_pred[c] = self.n_neighbors

            if n_pred:
                # KNN index for the samples predicted into class `c`
                self.index_knn_pred[c] = KNNIndex(
                    features[ind, :], n_neighbors=self.n_neighbors_pred[c],
                    metric=self.metric, metric_kwargs=self.metric_kwargs,
                    shared_nearest_neighbors=self.shared_nearest_neighbors,
                    approx_nearest_neighbors=self.approx_nearest_neighbors,
                    n_jobs=self.n_jobs,
                    low_memory=self.low_memory,
                    seed_rng=self.seed_rng
                )
                # Nearest-neighbor distance to the samples predicted into class `c`
                _, nn_distance = self.index_knn_pred[c].query_self(k=1)
                self.distance_ratio_train[ind, 0] = nn_distance[:, 0]
            else:
                raise ValueError("No predicted samples from class '{}'. Cannot proceed.".format(c))

            # Labeled samples from class `c`
            ind = np.where(labels == c)[0]
            self.indices_true[c] = ind
            n_true = ind.shape[0]
            # Number of neighbors for the labeled samples from class `c`
            if set_n_neighbors:
                self.n_neighbors_true[c] = int(np.ceil(n_true ** self.neighborhood_constant))
            else:
                self.n_neighbors_true[c] = self.n_neighbors

            if n_true:
                # KNN index for the labeled samples from class `c`
                self.index_knn_true[c] = KNNIndex(
                    features[ind, :], n_neighbors=self.n_neighbors_true[c],
                    metric=self.metric, metric_kwargs=self.metric_kwargs,
                    shared_nearest_neighbors=self.shared_nearest_neighbors,
                    approx_nearest_neighbors=self.approx_nearest_neighbors,
                    n_jobs=self.n_jobs,
                    low_memory=self.low_memory,
                    seed_rng=self.seed_rng
                )
                # Nearest-neighbor distance to the labeled samples from class `c`
                _, nn_distance = self.index_knn_true[c].query_self(k=1)
                self.distance_ratio_train[ind, i + 1] = nn_distance[:, 0]
            else:
                raise ValueError("No labeled samples from class '{}'. Cannot proceed.".format(c))

        # Loop over each class and calculate the denominator of the distance ratio in the trust score
        for i, c in enumerate(self.labels_unique):
            # Samples predicted into class `c`
            ind1 = self.indices_pred[c]
            features_temp1 = features[ind1, :]
            dist_temp1 = np.zeros((ind1.shape[0], self.n_classes - 1))
            # Labeled samples from class `c`
            ind2 = self.indices_true[c]
            features_temp2 = features[ind2, :]
            dist_temp2 = np.zeros((ind2.shape[0], self.n_classes - 1))
            j = 0
            # Loop over every class other than `c`
            for k in self.labels_unique:
                if k == c:
                    continue

                # Nearest-neighbor distance from the set of samples predicted into class `k`
                _, nn_distance = self.index_knn_pred[k].query(features_temp1, k=1)
                dist_temp1[:, j] = nn_distance[:, 0]
                # Nearest-neighbor distance from the set of labeled samples from class `k`
                _, nn_distance = self.index_knn_true[k].query(features_temp2, k=1)
                dist_temp2[:, j] = nn_distance[:, 0]
                j += 1

            # Distance ratio for the samples predicted into class `c`
            v = np.clip(np.min(dist_temp1, axis=1), sys.float_info.epsilon, None)
            self.distance_ratio_train[ind1, 0] = self.distance_ratio_train[ind1, 0] / v
            # Distance ratio for the labeled samples from class `c`
            v = np.clip(np.min(dist_temp2, axis=1), sys.float_info.epsilon, None)
            self.distance_ratio_train[ind2, i + 1] = self.distance_ratio_train[ind2, i + 1] / v

        # Calculate the scores and p-values for each samples
        self.scores_train, p_values = self.score(features, labels_pred, is_train=True, bootstrap=bootstrap)
        return self.scores_train, p_values

    def score(self, features_test, labels_pred_test, is_train=False, log_transform=True, bootstrap=True):
        """
        Given the test feature vectors and their corresponding predicted labels, calculate a vector of scores for
        each test sample. Set `is_train = True` only if the `fit` method was called using `features_test`.

        :param features_test: numpy array of shape `(N, d)` where `N` and `d` are the number of samples and
                              dimension respectively.
        :param labels_pred_test: numpy array of shape `(N, )` with the predicted labels per sample.
        :param is_train: Set to True if points from `features_test` were used for training, i.e. by the fit method.
        :param log_transform: Set to True to apply negative log transformation to the p-values.
        :param bootstrap: Set to True to calculate a bootstrap resampled estimate of the p-value.
        :return: (scores, p_values)
            scores: numpy array of shape `(N, m + 1)` with a vector of scores for each sample, where `m` is the
                    number of classes. The first column `scores[:, 0]` gives the scores conditioned on the predicted
                    class. The remaining columns `scores[:, i]` for `i = 1, . . ., m` gives the scores conditioned
                    on `i - 1` being the candidate true class for the test sample.
            p_values: numpy array of same shape as `scores` containing the negative-log-transformed empirical
                      p-values of the scores.
        """
        n_test = labels_pred_test.shape[0]
        if not is_train:
            if self.scaler is not None:
                features_test = self.scaler.transform(features_test)

        scores = np.zeros((n_test, 1 + self.n_classes))
        p_values = np.zeros((n_test, 1 + self.n_classes))
        preds_unique = self.labels_unique if (n_test > 1) else [labels_pred_test[0]]
        cnt_par = 0
        for c_hat in preds_unique:
            # Index of samples predicted into class `c_hat`
            ind = np.where(labels_pred_test == c_hat)[0]
            if ind.shape[0]:
                if not is_train:
                    temp_arr = features_test[ind, :]
                    # Nearest-neighbor distance to the samples predicted into class `c_hat`
                    _, nn_distance = self.index_knn_pred[c_hat].query(temp_arr, k=1)
                    scores[ind, 0] = nn_distance[:, 0]

                    dist_temp = np.zeros((ind.shape[0], self.n_classes - 1))
                    j = 0
                    # Loop over every class other than `c_hat`
                    for k in self.labels_unique:
                        if k == c_hat:
                            continue

                        # Nearest-neighbor distance to the samples predicted into class `k`
                        _, nn_distance = self.index_knn_pred[k].query(temp_arr, k=1)
                        dist_temp[:, j] = nn_distance[:, 0]
                        j += 1

                    # Distance ratio
                    v = np.clip(np.min(dist_temp, axis=1), sys.float_info.epsilon, None)
                    scores[ind, 0] = scores[ind, 0] / v
                    # p-value of the distance ratio
                    p_values[ind, 0] = pvalue_score(
                        self.scores_train[self.indices_pred[c_hat], 0], scores[ind, 0], log_transform=log_transform,
                        bootstrap=bootstrap
                    )
                else:
                    # Precomputed distance ratios
                    scores[ind, 0] = self.distance_ratio_train[ind, 0]
                    # p-value of the distance ratio
                    p_values[ind, 0] = pvalue_score(
                        scores[ind, 0], scores[ind, 0], log_transform=log_transform, bootstrap=bootstrap
                    )

                cnt_par += ind.shape[0]
                if cnt_par >= n_test:
                    break

        for i, c in enumerate(self.labels_unique):
            if not is_train:
                # Nearest-neighbor distance to the labeled samples from class `c`
                _, nn_distance = self.index_knn_true[c].query(features_test, k=1)
                scores[:, i + 1] = nn_distance[:, 0]

                dist_temp = np.zeros((n_test, self.n_classes - 1))
                j = 0
                # Loop over every class other than `c`
                for k in self.labels_unique:
                    if k == c:
                        continue

                    # Nearest-neighbor distance to the labeled samples from class `k`
                    _, nn_distance = self.index_knn_true[k].query(features_test, k=1)
                    dist_temp[:, j] = nn_distance[:, 0]
                    j += 1

                # Distance ratio
                v = np.clip(np.min(dist_temp, axis=1), sys.float_info.epsilon, None)
                scores[:, i + 1] = scores[:, i + 1] / v
                # p-value of the distance ratio
                p_values[:, i + 1] = pvalue_score(
                    self.scores_train[self.indices_true[c], i + 1], scores[:, i + 1], log_transform=log_transform,
                    bootstrap=bootstrap
                )
            else:
                # Precomputed distance ratios
                scores[:, i + 1] = self.distance_ratio_train[:, i + 1]
                # p-value of the distance ratio
                p_values[:, i + 1] = pvalue_score(
                    scores[self.indices_true[c], i + 1], scores[:, i + 1], log_transform=log_transform,
                    bootstrap=bootstrap
                )

        return scores, p_values


class LLEScore(TestStatistic):
    """
    Class that calculates the locally linear embedding (LLE) reconstruction error norm as the test statistic for
    any layer embedding.

    NOTE: For this method, do not apply dimension reduction to the layer embeddings. We want to estimate the LLE
    reconstruction error in the original layer embedding space.
    """
    def __init__(self, **kwargs):
        super(LLEScore, self).__init__(
            neighborhood_constant=kwargs.get('neighborhood_constant', NEIGHBORHOOD_CONST),
            n_neighbors=kwargs.get('n_neighbors', None),
            metric=kwargs.get('metric', METRIC_DEF),
            metric_kwargs=kwargs.get('metric_kwargs', None),
            shared_nearest_neighbors=False,     # Intentionally set to False
            approx_nearest_neighbors=kwargs.get('approx_nearest_neighbors', True),
            n_jobs=kwargs.get('n_jobs', 1),
            low_memory=kwargs.get('low_memory', False),
            seed_rng=kwargs.get('seed_rng', SEED_DEFAULT)
        )

        # Feature means and PCA transformation matrix
        self.mean_data = None
        self.transform_pca = None
        self.reg_eps = 0.001
        # Number of neighbors can be set based on the number of samples per class if `self.n_neighbors = None`
        self.n_neighbors_pred = dict()
        self.n_neighbors_true = dict()
        # KNN indices for the data points from each predicted class and true class
        self.index_knn_pred = dict()
        self.index_knn_true = dict()
        # Norm of the LLE reconstruction errors for the training data
        self.errors_lle_train = None
        # Scores for the training data
        self.scores_train = None
        # Index of train samples from each class based on the true class and predicted class
        self.indices_true = dict()
        self.indices_pred = dict()
        # Feature vectors used to build the KNN graph for each predicted class and true class
        self.features_knn_pred = dict()
        self.features_knn_true = dict()

    def fit(self, features, labels, labels_pred, labels_unique=None, bootstrap=False,
            min_dim_pca=10000, pca_cutoff=PCA_CUTOFF, reg_eps=0.001):
        """
        Use the given feature vectors, true labels, and predicted labels to estimate the scoring model.

        :param features: numpy array of shape `(N, d)` where `N` and `d` are the number of samples and
                         dimension respectively.
        :param labels: numpy array of shape `(N, )` with the true labels per sample.
        :param labels_pred: numpy array of shape `(N, )` with the predicted labels per sample.
        :param labels_unique: None or a numpy array with the unique labels. For example, np.arange(1, 11). This can
                              be supplied as input during repeated calls to avoid having the find the unique
                              labels each time.
        :param bootstrap: Set to True in order to calculate a bootstrap resampled estimate of the p-value.
                          The default value is False because the p-values returned by the fit method are usually not
                          used down the line. Not using the bootstrap her makes it faster.
        :param min_dim_pca: (int) minimum dimension above which PCA pre-processing is applied.
        :param pca_cutoff: float value in (0, 1] specifying the proportion of cumulative data variance to preserve
                           in the projected (dimension-reduced) data.
        :param reg_eps: small float value that multiplies the trace to regularize the Gram matrix, if it is
                        close to singular.

        :return: (scores, p_values)
            scores: numpy array of shape `(N, m + 1)` with a vector of scores for each sample, where `m` is the
                    number of classes. The first column `scores[:, 0]` gives the scores conditioned on the predicted
                    class. The remaining columns `scores[:, i]` for `i = 1, . . ., m` gives the scores conditioned
                    on `i - 1` being the candidate true class for the test sample.
            p_values: numpy array of same shape as `scores` containing the negative-log-transformed empirical
                      p-values of the scores.
        """
        set_n_neighbors = True if (self.n_neighbors is None) else False
        # `fit` method of the super class
        super(LLEScore, self).fit(features, labels, labels_pred, labels_unique=labels_unique)
        self.reg_eps = reg_eps

        if self.dim >= min_dim_pca:
            logger.info("Data dimension = {:d}. Applying PCA as a pre-processing step.".format(self.dim))
            features, self.mean_data, self.transform_pca = pca_wrapper(
                features, cutoff=pca_cutoff, seed_rng=self.seed_rng
            )
            self.dim = features.shape[1]

        # Column 0 corresponds to the LLE reconstruction errors conditioned on the predicted class.
        # Column `i` for `i = 1, 2, . . .` correspond to the LLE reconstruction errors conditioned on the true
        # class being `i - 1`
        self.errors_lle_train = np.zeros((self.n_train, 1 + self.n_classes))

        logger.info("Building KNN indices for nearest neighbor queries from each predicted and each true class.")
        for i, c in enumerate(self.labels_unique):
            logger.info("Processing class {}:".format(c))
            # Samples predicted into class `c`
            ind = np.where(labels_pred == c)[0]
            self.indices_pred[c] = ind
            n_pred = ind.shape[0]
            # Number of neighbors for the samples predicted into class `c`
            if set_n_neighbors:
                self.n_neighbors_pred[c] = int(np.ceil(n_pred ** self.neighborhood_constant))
            else:
                self.n_neighbors_pred[c] = self.n_neighbors

            # Number of neighbors should not exceed the number of dimensions in order to avoid a singular gram matrix
            if self.n_neighbors_pred[c] >= self.dim:
                self.n_neighbors_pred[c] = max(self.dim - 1, 1)

            if n_pred:
                # KNN index for the samples predicted into class `c`
                self.features_knn_pred[c] = features[ind, :]
                self.index_knn_pred[c] = KNNIndex(
                    self.features_knn_pred[c], n_neighbors=self.n_neighbors_pred[c],
                    metric=self.metric, metric_kwargs=self.metric_kwargs,
                    shared_nearest_neighbors=self.shared_nearest_neighbors,
                    approx_nearest_neighbors=self.approx_nearest_neighbors,
                    n_jobs=self.n_jobs,
                    low_memory=self.low_memory,
                    seed_rng=self.seed_rng
                )
                # # LLE reconstruction errors of the points from `self.features_knn_pred[c]`
                nn_indices, _ = self.index_knn_pred[c].query_self(k=self.n_neighbors_pred[c])
                self.errors_lle_train[ind, 0] = self._calc_reconstruction_errors(
                    self.features_knn_pred[c], self.features_knn_pred[c], nn_indices
                )
            else:
                raise ValueError("No predicted samples from class '{}'. Cannot proceed.".format(c))

            # Labeled samples from class `c`
            ind = np.where(labels == c)[0]
            self.indices_true[c] = ind
            n_true = ind.shape[0]
            # Number of neighbors for the labeled samples from class `c`
            if set_n_neighbors:
                self.n_neighbors_true[c] = int(np.ceil(n_true ** self.neighborhood_constant))
            else:
                self.n_neighbors_true[c] = self.n_neighbors

            # Number of neighbors should not exceed the number of dimensions in order to avoid a singular gram matrix
            if self.n_neighbors_true[c] >= self.dim:
                self.n_neighbors_true[c] = max(self.dim - 1, 1)

            if n_true:
                # KNN index for the labeled samples from class `c`
                self.features_knn_true[c] = features[ind, :]
                self.index_knn_true[c] = KNNIndex(
                    self.features_knn_true[c], n_neighbors=self.n_neighbors_true[c],
                    metric=self.metric, metric_kwargs=self.metric_kwargs,
                    shared_nearest_neighbors=self.shared_nearest_neighbors,
                    approx_nearest_neighbors=self.approx_nearest_neighbors,
                    n_jobs=self.n_jobs,
                    low_memory=self.low_memory,
                    seed_rng=self.seed_rng
                )
                # LLE reconstruction errors for the labeled samples from class `c`
                nn_indices, _ = self.index_knn_true[c].query_self(k=self.n_neighbors_true[c])
                self.errors_lle_train[ind, i + 1] = self._calc_reconstruction_errors(
                    self.features_knn_true[c], self.features_knn_true[c], nn_indices
                )
                # Commenting this block out to avoid unneccessary computation.
                # The reconstruction errors for these samples are not used anywhere else.
                '''
                # LLE reconstruction errors for the labeled samples not from class `c`
                ind_comp = np.where(labels != c)[0]
                temp_arr = features[ind_comp, :]
                nn_indices, _ = self.index_knn_true[c].query(temp_arr, k=self.n_neighbors_true[c])
                self.errors_lle_train[ind_comp, i + 1] = self._calc_reconstruction_errors(
                    temp_arr, self.features_knn_true[c], nn_indices
                )
                '''
            else:
                raise ValueError("No labeled samples from class '{}'. Cannot proceed.".format(c))

        # Calculate the scores and p-values for each samples
        self.scores_train, p_values = self.score(features, labels_pred, is_train=True, bootstrap=bootstrap)
        return self.scores_train, p_values

    def score(self, features_test, labels_pred_test, is_train=False, log_transform=True, bootstrap=True):
        """
        Given the test feature vectors and their corresponding predicted labels, calculate a vector of scores for
        each test sample. Set `is_train = True` only if the `fit` method was called using `features_test`.

        :param features_test: numpy array of shape `(N, d)` where `N` and `d` are the number of samples and
                              dimension respectively.
        :param labels_pred_test: numpy array of shape `(N, )` with the predicted labels per sample.
        :param is_train: Set to True if points from `features_test` were used for training, i.e. by the fit method.
        :param log_transform: Set to True to apply negative log transformation to the p-values.
        :param bootstrap: Set to True to calculate a bootstrap resampled estimate of the p-value.

        :return: (scores, p_values)
            scores: numpy array of shape `(N, m + 1)` with a vector of scores for each sample, where `m` is the
                    number of classes. The first column `scores[:, 0]` gives the scores conditioned on the predicted
                    class. The remaining columns `scores[:, i]` for `i = 1, . . ., m` gives the scores conditioned
                    on `i - 1` being the candidate true class for the test sample.
            p_values: numpy array of same shape as `scores` containing the negative-log-transformed empirical
                      p-values of the scores.
        """
        n_test = labels_pred_test.shape[0]
        if (self.transform_pca is not None) and (not is_train):
            # Apply a PCA transformation to the test features
            features_test = np.dot(features_test - self.mean_data, self.transform_pca)

        scores = np.zeros((n_test, 1 + self.n_classes))
        p_values = np.zeros((n_test, 1 + self.n_classes))
        preds_unique = self.labels_unique if (n_test > 1) else [labels_pred_test[0]]
        cnt_par = 0
        for c_hat in preds_unique:
            # Index of samples predicted into class `c_hat`
            ind = np.where(labels_pred_test == c_hat)[0]
            if ind.shape[0]:
                if not is_train:
                    # LLE reconstruction errors relative to the samples predicted into class `c_hat`
                    temp_arr = features_test[ind, :]
                    nn_indices, _ = self.index_knn_pred[c_hat].query(temp_arr, k=self.n_neighbors_pred[c_hat])
                    scores[ind, 0] = self._calc_reconstruction_errors(
                        temp_arr, self.features_knn_pred[c_hat], nn_indices
                    )
                    # p-value of the LLE reconstruction errors
                    p_values[ind, 0] = pvalue_score(
                        self.scores_train[self.indices_pred[c_hat], 0], scores[ind, 0], log_transform=log_transform,
                        bootstrap=bootstrap
                    )
                else:
                    # Precomputed LLE reconstruction errors
                    scores[ind, 0] = self.errors_lle_train[ind, 0]
                    # p-value of the LLE reconstruction errors
                    p_values[ind, 0] = pvalue_score(
                        scores[ind, 0], scores[ind, 0], log_transform=log_transform, bootstrap=bootstrap
                    )

                cnt_par += ind.shape[0]
                if cnt_par >= n_test:
                    break

        for i, c in enumerate(self.labels_unique):
            if not is_train:
                # LLE reconstruction errors relative to the samples labeled as class `c`
                nn_indices, _ = self.index_knn_true[c].query(features_test, k=self.n_neighbors_true[c])
                scores[:, i + 1] = self._calc_reconstruction_errors(
                    features_test, self.features_knn_true[c], nn_indices
                )
                # p-value of the LLE reconstruction errors
                p_values[:, i + 1] = pvalue_score(
                    self.scores_train[self.indices_true[c], i + 1], scores[:, i + 1], log_transform=log_transform,
                    bootstrap=bootstrap
                )
            else:
                # Precomputed LLE reconstruction errors
                scores[:, i + 1] = self.errors_lle_train[:, i + 1]
                # p-value of the LLE reconstruction errors
                p_values[:, i + 1] = pvalue_score(
                    scores[self.indices_true[c], i + 1], scores[:, i + 1], log_transform=log_transform,
                    bootstrap=bootstrap
                )

        return scores, p_values

    def _calc_reconstruction_errors(self, features, features_knn, nn_indices):
        """
        Calculate the norm of the reconstruction errors of the LLE embedding in the neighborhood of each point.

        :param features: numpy array of shape `(N, d)` where `N` and `d` are the number of samples and
                         dimension respectively.
        :param features_knn: numpy array of shape `(m, d)` with the feature vectors used for the original KNN
                             graph construction.
        :param nn_indices: numpy array of shape `(N, k)` where `k` is the number of neighbors.
        :return:
            - array with the norm of the reconstruction errors. Has shape `(N, )`.
        """
        N = nn_indices.shape[0]
        if self.n_jobs == 1:
            w = [helper_reconstruction_error(features, features_knn, nn_indices, self.reg_eps, i) for i in range(N)]
        else:
            helper_partial = partial(helper_reconstruction_error, features, features_knn, nn_indices, self.reg_eps)
            pool_obj = multiprocessing.Pool(processes=self.n_jobs)
            w = []
            _ = pool_obj.map_async(helper_partial, range(N), callback=w.extend)
            pool_obj.close()
            pool_obj.join()

        return np.array(w)
