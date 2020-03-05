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
from helpers.knn_index import KNNIndex
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
from helpers.utils import get_num_jobs
from helpers.constants import (
    NEIGHBORHOOD_CONST,
    SEED_DEFAULT,
    METRIC_DEF
)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def helper_pvalue(scores_null_repl, scores_obs, n_samp, i):
    scores_null = scores_null_repl[i, :]
    mask = scores_null[:, np.newaxis] >= scores_obs
    return mask.sum(axis=0, dtype=np.float) / n_samp


# Not using this version since it's slower than the numba version
def pvalue_score_alt(scores_null, scores_obs, log_transform=False, bootstrap=True, n_bootstrap=100, n_jobs=1):
    """
    Calculate the empirical p-values of the observed scores `scores_obs` with respect to the scores from the
    null distribution `scores_null`. Bootstrap resampling can be used to get better estimates of the p-values.

    :param scores_null: numpy array of shape `(m, )` with the scores from the null distribution.
    :param scores_obs: numpy array of shape `(n, )` with the observed scores.
    :param log_transform: set to True to apply negative log transform to the p-values.
    :param bootstrap: Set to True to calculate a bootstrap resampled estimate of the p-value.
    :param n_bootstrap: number of bootstrap resamples to use.
    :param n_jobs: number of jobs to execute in parallel.

    :return: numpy array with the p-values or negative-log-transformed p-values. Has the same shape as `scores_obs`.
    """
    n_samp = scores_null.shape[0]
    scores_obs = scores_obs[np.newaxis, :]
    mask = scores_null[:, np.newaxis] >= scores_obs
    p = mask.sum(axis=0, dtype=np.float) / n_samp
    if bootstrap:
        scores_null_repl = np.random.choice(scores_null, size=(n_bootstrap, n_samp), replace=True)
        if n_jobs == 1:
            p_repl = [helper_pvalue(scores_null_repl, scores_obs, n_samp, i) for i in range(n_bootstrap)]
        else:
            helper_partial = partial(helper_pvalue, scores_null_repl, scores_obs, n_samp)
            pool_obj = multiprocessing.Pool(processes=n_jobs)
            p_repl = []
            _ = pool_obj.map_async(helper_partial, range(n_bootstrap), callback=p_repl.extend)
            pool_obj.close()
            pool_obj.join()

        p_repl.append(p)
        # Average p-value from the bootstrap replications
        p = np.mean(np.array(p_repl), axis=0)

    p = np.clip(p, sys.float_info.epsilon, None)
    if log_transform:
        return -np.log(p)
    else:
        return p


# numba version
@njit(parallel=True)
def pvalue_score(scores_null, scores_obs, log_transform=False, bootstrap=True, n_bootstrap=100):
    """
    Calculate the empirical p-values of the observed scores `scores_obs` with respect to the scores from the
    null distribution `scores_null`. Bootstrap resampling can be used to get better estimates of the p-values.

    :param scores_null: numpy array of shape `(m, )` with the scores from the null distribution.
    :param scores_obs: numpy array of shape `(n, )` with the observed scores.
    :param log_transform: set to True to apply negative log transform to the p-values.
    :param bootstrap: Set to True to calculate a bootstrap resampled estimate of the p-value.
    :param n_bootstrap: number of bootstrap resamples to use.
    :param n_jobs: number of jobs to execute in parallel.

    :return: numpy array with the p-values or negative-log-transformed p-values. Has the same shape as `scores_obs`.
    """
    eps = 1e-16
    n_samp = float(scores_null.shape[0])
    n_obs = scores_obs.shape[0]
    p = np.zeros(n_obs)
    for i in prange(n_obs):
        mask = scores_null >= scores_obs[i]
        p[i] = mask[mask].shape[0] / n_samp

    if bootstrap:
        scores_null_repl = np.random.choice(scores_null, size=(n_bootstrap, scores_null.shape[0]), replace=True)
        p_sum = p
        for b in prange(n_bootstrap):
            scores_null_curr = scores_null_repl[b, :]
            p_curr = np.zeros(n_obs)
            for i in prange(n_obs):
                mask = scores_null_curr >= scores_obs[i]
                p_curr[i] = mask[mask].shape[0] / n_samp

            p_sum += p_curr

        # Average p-value from the bootstrap replications
        p = p_sum / (n_bootstrap + 1.)

    p[p < eps] = eps
    if log_transform:
        return -np.log(p)
    else:
        return p


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

        :return: (scores, p_values)
            scores: numpy array of shape `(N, m + 1)` with a vector of scores for each sample, where `m` is the
                    number of classes. The first column `scores[:, 0]` gives the scores conditioned on the predicted
                    class. The remaining columns `scores[:, i]` for `i = 1, . . ., m` gives the scores conditioned
                    on `i - 1` being the candidate true class for the test sample.
            p_values: numpy array of same shape as `scores` containing the negative-log-transformed empirical
                      p-values of the scores.
        """
        super(MultinomialScore, self).fit(features, labels, labels_pred, labels_unique=labels_unique)

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

        # Parameter estimation conditioned on the predicted class and the true class
        mat_ones = np.ones((self.n_classes, self.n_classes))
        self.proba_params_pred = (1. / self.n_classes) * mat_ones
        self.proba_params_true = (1. / self.n_classes) * mat_ones
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

        # Calculate the scores and p-values for each sample. Not using bootstrap because the p-values calculated
        # here are not used
        self.scores_train, p_values = self.score(features, labels_pred, is_train=True, bootstrap=False)
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
                # Likelihood ratio statistic conditioned on the predicted class `c_hat`
                scores[ind, 0] = self.multinomial_lrt(data_counts[ind, :], self.proba_params_pred[i, :],
                                                      self.n_neighbors)
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
            # Likelihood ratio statistic conditioned on the candidate true class `c`
            scores[:, i + 1] = self.multinomial_lrt(data_counts, self.proba_params_true[i, :], self.n_neighbors)
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
    def multinomial_lrt(data_counts, proba_params, n_neighbors):
        mat = data_counts * (np.log(np.clip(data_counts, 1e-16, None)) - np.log(n_neighbors * proba_params))
        return np.sum(mat, axis=1)


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

        # Calculate the scores and p-values for each sample. Not using bootstrap because the p-values estimated
        # here are not used
        self.scores_train, p_values = self.score(features, labels_pred, is_train=True, bootstrap=False)
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
            metric='euclidean',     # Has to be 'euclidean' for LID estimation
            shared_nearest_neighbors=False,     # Intentionally set to False
            approx_nearest_neighbors=kwargs.get('approx_nearest_neighbors', True),
            n_jobs=kwargs.get('n_jobs', 1),
            low_memory=kwargs.get('low_memory', False),
            seed_rng=kwargs.get('seed_rng', SEED_DEFAULT)
        )

        self.lid_estimates_train = None
        # Median LID estimate of the samples predicted into each class
        self.lid_median_pred = None
        # Median LID estimate of the samples labeled from each class
        self.lid_median_true = None
        # Scores for the training data
        self.scores_train = None
        # Index of train samples from each class based on the true class and predicted class
        self.indices_true = dict()
        self.indices_pred = dict()

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

        :return: (scores, p_values)
            scores: numpy array of shape `(N, m + 1)` with a vector of scores for each sample, where `m` is the
                    number of classes. The first column `scores[:, 0]` gives the scores conditioned on the predicted
                    class. The remaining columns `scores[:, i]` for `i = 1, . . ., m` gives the scores conditioned
                    on `i - 1` being the candidate true class for the test sample.
            p_values: numpy array of same shape as `scores` containing the negative-log-transformed empirical
                      p-values of the scores.
        """
        super(LIDScore, self).fit(features, labels, labels_pred, labels_unique=labels_unique)

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
        # Indices and distances of the nearest neighbors of the points from `features`
        _, nn_distances = self.index_knn.query_self(k=self.n_neighbors)

        logger.info("Calculating the local intrinsic dimension estimates from the k nearest neighbor distances "
                    "of each sample.")
        self.lid_estimates_train = lid_mle_amsaleg(nn_distances)

        self.lid_median_pred = np.ones(self.n_classes)
        self.lid_median_true = np.ones(self.n_classes)
        for i, c in enumerate(self.labels_unique):
            # LID values of samples predicted into class `c`
            ind = np.where(labels_pred == c)[0]
            self.indices_pred[c] = ind
            if ind.shape[0]:
                self.lid_median_pred[i] = np.median(self.lid_estimates_train[ind])
            else:
                logger.warning("No samples are predicted into class '{}'. Setting the median LID "
                               "value to 1.".format(c))

            # LID values of samples labeled as class `c`
            ind = np.where(labels == c)[0]
            self.indices_true[c] = ind
            if ind.shape[0]:
                self.lid_median_true[i] = np.median(self.lid_estimates_train[ind])
            else:
                logger.warning("No labeled samples from class '{}'. Setting the median LID value to 1.".format(c))

        # Calculate the scores and p-values for each sample. Not using bootstrap because the p-values calculated
        # here are not used
        self.scores_train, p_values = self.score(features, labels_pred, is_train=True, bootstrap=False)
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
        # Query the index of `self.n_neighbors` nearest neighbors of each test sample and find the LID estimates
        if is_train:
            lid_estimates = self.lid_estimates_train
        else:
            _, nn_distances = self.index_knn.query(features_test, k=self.n_neighbors)
            lid_estimates = lid_mle_amsaleg(nn_distances)

        scores = np.zeros((n_test, 1 + self.n_classes))
        p_values = np.zeros((n_test, 1 + self.n_classes))
        preds_unique = self.labels_unique if (n_test > 1) else [labels_pred_test[0]]
        cnt_par = 0
        for c_hat in preds_unique:
            i = self.label_encoder([c_hat])[0]
            # Index of samples predicted into class `c_hat`
            ind = np.where(labels_pred_test == c_hat)[0]
            if ind.shape[0]:
                # LID scores normalized by the median LID value for the samples predicted into class `c`
                # scores[ind, 0] = lid_estimates[ind] / self.lid_median_pred[i]
                scores[ind, 0] = lid_estimates[ind]
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
            # LID scores normalized by the median LID value for the samples labeled as class `c`
            # scores[:, i + 1] = lid_estimates / self.lid_median_true[i]
            scores[:, i + 1] = lid_estimates
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

        self.mean_data = None
        self.transform_pca = None
        self.reg_eps = 0.001
        # Norm of the LLE reconstruction errors for the training data
        self.errors_lle_train = None
        # Median reconstruction error of the samples predicted into each class
        self.err_median_pred = None
        # Median reconstruction error of the labeled samples from each class
        self.err_median_true = None
        # Scores for the training data
        self.scores_train = None
        # Index of train samples from each class based on the true class and predicted class
        self.indices_true = dict()
        self.indices_pred = dict()
        # Feature vectors used to build the KNN graph
        self.features_knn = None

    def fit(self, features, labels, labels_pred, labels_unique=None,
            min_dim_pca=1000, pca_cutoff=0.995, reg_eps=0.001):
        """
        Use the given feature vectors, true labels, and predicted labels to estimate the scoring model.

        :param features: numpy array of shape `(N, d)` where `N` and `d` are the number of samples and
                         dimension respectively.
        :param labels: numpy array of shape `(N, )` with the true labels per sample.
        :param labels_pred: numpy array of shape `(N, )` with the predicted labels per sample.
        :param labels_unique: None or a numpy array with the unique labels. For example, np.arange(1, 11). This can
                              be supplied as input during repeated calls to avoid having the find the unique
                              labels each time.
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
        super(LLEScore, self).fit(features, labels, labels_pred, labels_unique=labels_unique)
        self.reg_eps = reg_eps

        if features.shape[1] >= min_dim_pca:
            logger.info("Applying PCA as first-level dimension reduction step.")
            features, self.mean_data, self.transform_pca = pca_wrapper(
                features, cutoff=pca_cutoff, seed_rng=self.seed_rng
            )

        # If `self.neighbors > features.shape[1]` (number of neighbors larger than the data dimension), then the
        # Gram matrix that comes up while solving for the neighborhood weights becomes singular. To avoid this,
        # we set `self.neighbors = features.shape[1]` or add a small nonzero value to the diagonal elements of the
        # Gram matrix
        d = features.shape[1]
        if self.n_neighbors >= d:
            k = max(d - 1, 1)
            logger.info("Reducing the number of neighbors from {:d} to {:d} to avoid singular Gram "
                        "matrix while solving for neighborhood weights.".format(self.n_neighbors, k))
            self.n_neighbors = k

        logger.info("Building a KNN index for nearest neighbor queries.")
        self.features_knn = features
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

        logger.info("Calculating the optimal LLE reconstruction from the nearest neighbors of each sample.")
        self.errors_lle_train = self._calc_reconstruction_errors(features, self.features_knn, nn_indices)

        self.err_median_pred = np.ones(self.n_classes)
        self.err_median_true = np.ones(self.n_classes)
        for i, c in enumerate(self.labels_unique):
            # Reconstruction error of samples predicted into class `c`
            ind = np.where(labels_pred == c)[0]
            self.indices_pred[c] = ind
            if ind.shape[0]:
                self.err_median_pred[i] = np.clip(np.median(self.errors_lle_train[ind]), 1e-16, None)
            else:
                logger.warning("No samples are predicted into class '{}'. Setting the median reconstruction error "
                               "to 1.".format(c))

            # Reconstruction error of samples labeled as class `c`
            ind = np.where(labels == c)[0]
            self.indices_true[c] = ind
            if ind.shape[0]:
                self.err_median_true[i] = np.clip(np.median(self.errors_lle_train[ind]), 1e-16, None)
            else:
                logger.warning("No labeled samples from class '{}'. Setting the median reconstruction error "
                               "to 1.".format(c))

        # Calculate the scores and p-values for each sample. Not using bootstrap because the p-values calculated
        # here are not used
        self.scores_train, p_values = self.score(features, labels_pred, is_train=True, bootstrap=False)
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
        if is_train:
            errors_lle = self.errors_lle_train
        else:
            # Apply a PCA transformation to the test features if required
            if self.transform_pca is not None:
                features_test = np.dot(features_test - self.mean_data, self.transform_pca)

            # Query the index of `self.n_neighbors` nearest neighbors of each test sample
            nn_indices, nn_distances = self.index_knn.query(features_test, k=self.n_neighbors)

            # Find the optimal convex reconstruction of each point from its nearest neighbors and the norm of
            # the reconstruction errors
            errors_lle = self._calc_reconstruction_errors(features_test, self.features_knn, nn_indices)

        scores = np.zeros((n_test, 1 + self.n_classes))
        p_values = np.zeros((n_test, 1 + self.n_classes))
        preds_unique = self.labels_unique if (n_test > 1) else [labels_pred_test[0]]
        cnt_par = 0
        for c_hat in preds_unique:
            i = self.label_encoder([c_hat])[0]
            # Index of samples predicted into class `c_hat`
            ind = np.where(labels_pred_test == c_hat)[0]
            if ind.shape[0]:
                # Reconstruction error scores normalized by the median value for the samples predicted into class `c`
                # scores[ind, 0] = errors_lle[ind] / self.err_median_pred[i]
                scores[ind, 0] = errors_lle[ind]
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
            # Reconstruction error scores normalized by the median value for the samples labeled as class `c`
            # scores[:, i + 1] = errors_lle / self.err_median_true[i]
            scores[:, i + 1] = errors_lle
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
