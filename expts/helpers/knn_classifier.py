"""
Basic k nearest neighbors classifier that supports approximate nearest neighbor querying and custom distance
metrics including shared nearest neighbors.
"""
import numpy as np
import multiprocessing
import operator
from functools import partial
import logging
from knn_index import KNNIndex
from sklearn.model_selection import StratifiedKFold
from itertools import product
from numba import njit, int64, float64
from numba.types import Tuple
from dimension_reduction_methods import (
    pca_wrapper,
    wrapper_data_projection,
    METHODS_LIST
)
from utils import get_num_jobs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def knn_parameter_search(data, labels, k_range,
                         dim_proj_range=None, method_proj=None,
                         num_cv_folds=5,
                         metric='euclidean', metric_kwargs=None,
                         shared_nearest_neighbors=False,
                         approx_nearest_neighbors=True,
                         skip_preprocessing=False,
                         pca_cutoff=1.0,
                         n_jobs=-1,
                         seed_rng=123):
    """
    Search for the best value of `k` (number of neighbors) of a KNN classifier using cross-validation. Error rate
    is the metric. Optionally, you can also search over a range of reduced data dimensions via the parameters
    `dim_proj_range` and `method_proj`. In this case, the dimensionality reduction method `method_proj` is applied
    to reduce the dimension of the data to the specified values, and the search is doing over both `k` and the
    dimension.

    :param data: numpy array with the training data of shape `(N, d)`, where `N` is the number of samples
                 and `d` is the dimension.
    :param labels: numpy array with the training labels of shape `(N, )`.
    :param k_range: list or array with the k values to search over. Expected to be sorted in increasing values.
    :param dim_proj_range: None or a list/array with the dimension values to search over. Set to `None` if there
                           is no need to search over the data dimension.
    :param method_proj: None or a method for performing dimension reduction. The method string has to be one of the
                        defined values `['LPP', 'OLPP', 'NPP', 'ONPP', 'PCA']`.
    :param num_cv_folds: int value > 1 that specifies the number of cross-validation folds.
    :param metric: same as the function `wrapper_knn`.
    :param metric_kwargs: same as the function `wrapper_knn`.
    :param shared_nearest_neighbors: same as the function `wrapper_knn`.
    :param approx_nearest_neighbors: same as the function `wrapper_knn`.
    :param skip_preprocessing: Set to True to skip the pre-processing step using PCA to remove noisy features
                               with low variance.
    :param pca_cutoff: cumulative variance cutoff value in (0, 1]. This value is used for PCA.
    :param n_jobs: None or int value that specifies the number of parallel jobs. If set to None, -1, or 0, this will
                   use all the available CPU cores. If set to negative values, this value will be subtracted from
                   the available number of CPU cores. For example, `n_jobs = -2` will use `cpu_count - 2`.
    :param seed_rng: same as the function `wrapper_knn`.

    :return:
    (k_best, dim_best, error_rate_min, data_proj), where
        - k_best: selected best value for `k` from the list `k_range`.
        - dim_best: select best value of dimension. Can be ignored if no search is performed over the data dimension.
        - error_rate_min: minimum cross-validation error rate.
        - data_proj: projected (dimension reduced) data corresponding to the `dim_best`. Can be ignored if no search
                     is performed over the data dimension.
    """
    # Number of parallel jobs
    n_jobs = get_num_jobs(n_jobs)

    # Unique labels
    labels_unique = np.unique(labels)

    if skip_preprocessing:
        data_proj_list = [data]
        dim_proj_range = [data.shape[1]]
    elif method_proj is None:
        # Applying PCA as pre-processing step to remove noisy features
        data_proj, mean_data, transform_pca = pca_wrapper(data, cutoff=pca_cutoff, seed_rng=seed_rng)
        data_proj_list = [data_proj]
        dim_proj_range = [data_proj.shape[1]]
    else:
        if method_proj not in METHODS_LIST:
            raise ValueError("Invalid value '{}' specified for the argument 'method_proj'".format(method_proj))

        logger.info("Using {} for dimension reduction.".format(method_proj))
        if isinstance(dim_proj_range, int):
            dim_proj_range = [dim_proj_range]

        # Project the data to different reduced dimensions using the method `method_proj`
        data_proj_list = wrapper_data_projection(data, method_proj,
                                                 dim_proj=dim_proj_range,
                                                 metric=metric, metric_kwargs=metric_kwargs,
                                                 snn=shared_nearest_neighbors,
                                                 ann=approx_nearest_neighbors,
                                                 pca_cutoff=pca_cutoff,
                                                 n_jobs=n_jobs,
                                                 seed_rng=seed_rng)

    # Split the data into stratified folds for cross-validation
    skf = StratifiedKFold(n_splits=num_cv_folds, shuffle=True, random_state=seed_rng)
    nd = len(dim_proj_range)
    nk = len(k_range)
    if nd > 1:
        logger.info("Performing cross-validation to search for the best combination of number of neighbors and "
                    "projected data dimension:")
    else:
        logger.info("Performing cross-validation to search for the best number of neighbors:")

    error_rates_cv = np.zeros((nd, nk))
    for ind_tr, ind_te in skf.split(data, labels):
        # Each cv fold
        for i in range(nd):
            # Each projected dimension
            data_proj = data_proj_list[i]

            # KNN classifier model with the maximum k value in `k_range`
            knn_model = KNNClassifier(
                n_neighbors=k_range[-1],
                metric=metric, metric_kwargs=metric_kwargs,
                shared_nearest_neighbors=shared_nearest_neighbors,
                approx_nearest_neighbors=approx_nearest_neighbors,
                n_jobs=n_jobs,
                seed_rng=seed_rng
            )
            # Fit to the training data from this fold
            knn_model.fit(data_proj[ind_tr, :], labels[ind_tr], y_unique=labels_unique)

            # Get the label predictions for the different values of k in `k_range`.
            # `labels_test_pred` will be a numpy array of shape `(len(k_range), ind_te.shape[0])`
            labels_test_pred = knn_model.predict_multiple_k(data_proj[ind_te, :], k_range)

            # Error rate on the test data from this fold
            err_rate_fold = np.count_nonzero(labels_test_pred != labels[ind_te], axis=1) / float(ind_te.shape[0])
            error_rates_cv[i, :] = error_rates_cv[i, :] + err_rate_fold

    # Average cross-validated error rate
    error_rates_cv = error_rates_cv / num_cv_folds

    # Find the projected dimension and k value corresponding to the minimum error rate
    a = np.argmin(error_rates_cv)
    row_ind = np.repeat(np.arange(nd)[:, np.newaxis], nk, axis=1).ravel()
    col_ind = np.repeat(np.arange(nk)[np.newaxis, :], nd, axis=0).ravel()
    ir = row_ind[a]
    ic = col_ind[a]
    error_rate_min = error_rates_cv[ir, ic]
    k_best = k_range[ic]
    dim_best = dim_proj_range[ir]
    logger.info("Best value of k (number of neighbors) = {:d}. Data dimension = {:d}. "
                "Cross-validation error rate = {:.6f}".format(k_best, dim_best, error_rate_min))

    return k_best, dim_best, error_rate_min, data_proj_list[ir]


def wrapper_knn(data, labels, k,
                data_test=None, labels_test=None,
                metric='euclidean', metric_kwargs=None,
                shared_nearest_neighbors=False,
                approx_nearest_neighbors=True,
                n_jobs=1,
                seed_rng=123):
    """

    :param data: numpy array with the training data of shape `(N, d)`, where `N` is the number of samples
                 and `d` is the dimension.
    :param labels: numpy array with the training labels of shape `(N, )`.
    :param k: int value specifying the number of neighbors.
    :param data_test: None or a numpy array with the test data similar to `data`.
    :param labels_test: None or a numpy array with the test labels similar to `labels`.
    :param metric: predefined distance metric string or a callable that calculates a custom distance metric.
    :param metric_kwargs: None or a dict specifying any keyword arguments for the distance metric.
    :param shared_nearest_neighbors: Set to True in order to use the shared nearest neighbor (SNN) distance.
                                     This is a secondary distance metric that is found to be better suited to
                                     high dimensional data.
    :param approx_nearest_neighbors: Set to True in order to use an approximate nearest neighbor algorithm to
                                     find the nearest neighbors. This is recommended when the number of points is
                                     large and/or when the dimension of the data is high.
    :param n_jobs: Number of parallel jobs or processes. Set to -1 to use all the available cpu cores.
    :param seed_rng: int value specifying the seed for the random number generator.

    :return: error rate (in the range [0, 1]) on the test data (if provided as input) or the training data.
    """
    knn_model = KNNClassifier(
        n_neighbors=k,
        metric=metric, metric_kwargs=metric_kwargs,
        shared_nearest_neighbors=shared_nearest_neighbors,
        approx_nearest_neighbors=approx_nearest_neighbors,
        n_jobs=n_jobs,
        seed_rng=seed_rng
    )
    knn_model.fit(data, labels)

    if data_test is None:
        labels_pred = knn_model.predict(data, is_train=True)
        # error rate
        mask = labels_pred != labels
        err_rate = float(mask[mask].shape[0]) / labels.shape[0]
    else:
        labels_pred = knn_model.predict(data_test, is_train=False)
        # error rate
        mask = labels_pred != labels_test
        err_rate = float(mask[mask].shape[0]) / labels_test.shape[0]

    return err_rate


@njit(Tuple((float64[:], float64[:, :]))(int64[:, :], int64[:], int64), fastmath=True)
def neighbors_label_counts(index_neighbors, labels_train, n_classes):
    """
    Given the index of neighboring samples from the training set and the labels of the training samples,
    find the label counts among the k neighbors and assign the label corresponding to the highest count as the
    prediction.

    :param index_neighbors: numpy array of shape `(n, k)` with the index of `k` neighbors of `n` samples.
    :param labels_train: numpy array of shape `(m, )` with the class labels of the `m` training samples.
    :param n_classes: (int) number of distinct classes.

    :return:
        - labels_pred: numpy array of shape `(n, )` with the predicted labels of the `n` samples. Needs to converted
                       to type `np.int` at the calling function. Numba is not very flexible.
        - counts: numpy array of shape `(n, n_classes)` with the count of each class among the `k` neighbors.
    """
    n, k = index_neighbors.shape
    counts = np.zeros((n, n_classes))
    labels_pred = np.zeros(n)
    for i in range(n):
        cnt_max = -1.
        ind_max = 0
        for j in range(k):
            c = labels_train[index_neighbors[i, j]]
            counts[i, c] += 1
            if counts[i, c] > cnt_max:
                cnt_max = counts[i, c]
                ind_max = c

        labels_pred[i] = ind_max

    return labels_pred, counts


def helper_knn_predict(nn_indices, y_train, n_classes, label_dec, k):
    # Helper function for the class `KNNClassifier`. Could not make this a class method because it needs to
    # be serialized using `pickle` by `multiprocessing`.
    if k > 1:
        labels_pred, counts = neighbors_label_counts(nn_indices[:, :k], y_train, n_classes)
        labels_pred = labels_pred.astype(np.int)
    else:
        labels_pred = y_train[nn_indices[:, 0]]
        counts = np.ones(labels_pred.shape[0])

    return label_dec(labels_pred), counts


class KNNClassifier:
    """
    Basic k nearest neighbors classifier that supports approximate nearest neighbor querying and custom distance
    metrics including shared nearest neighbors.
    """
    def __init__(self,
                 n_neighbors=1,
                 metric='euclidean', metric_kwargs=None,
                 shared_nearest_neighbors=False,
                 approx_nearest_neighbors=True,
                 n_jobs=1,
                 seed_rng=123):
        """
        :param n_neighbors: int value specifying the number of nearest neighbors. Should be >= 1.
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
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.shared_nearest_neighbors = shared_nearest_neighbors
        self.approx_nearest_neighbors = approx_nearest_neighbors
        self.n_jobs = get_num_jobs(n_jobs)
        self.seed_rng = seed_rng

        self.index_knn = None
        self.y_train = None
        self.n_classes = None
        self.labels_dtype = None
        self.label_enc = None
        self.label_dec = None

    def fit(self, X, y, y_unique=None):
        """

        :param X: numpy array with the feature vectors of shape `(N, d)`, where `N` is the number of samples
                  and `d` is the dimension.
        :param y: numpy array of class labels of shape `(N, )`.
        :param y_unique: Allows the optional specification of the unique labels. Can be a tuple list, or numpy
                         array of the unique labels. If this is not specified, then it is found using
                         `numpy.unique`.
        :return: None
        """
        self.labels_dtype = y.dtype
        # Labels are mapped to dtype int because `numba` does not handle generic numpy arrays
        if y_unique is None:
            y_unique = np.unique(y)

        self.n_classes = len(y_unique)
        ind = np.arange(self.n_classes)
        # Mapping from label values to integers and its inverse
        d = dict(zip(y_unique, ind))
        self.label_enc = np.vectorize(d.__getitem__)

        d = dict(zip(ind, y_unique))
        self.label_dec = np.vectorize(d.__getitem__)
        self.y_train = self.label_enc(y)

        self.index_knn = KNNIndex(
            X,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            metric_kwargs=self.metric_kwargs,
            shared_nearest_neighbors=self.shared_nearest_neighbors,
            approx_nearest_neighbors=self.approx_nearest_neighbors,
            n_jobs=self.n_jobs,
            seed_rng=self.seed_rng
        )

    def predict(self, X, is_train=False):
        """
        Predict the class labels for the given inputs.

        :param X: numpy array with the feature vectors of shape `(N, d)`, where `N` is the number of samples
                  and `d` is the dimension.
        :param is_train: Set to True if prediction is being done on the same data used to train.

        :return: numpy array with the class predictions, of shape `(N, )`.
        """
        # Get the indices of the nearest neighbors from the training set
        nn_indices, nn_distances = self.index_knn.query(X, k=self.n_neighbors, exclude_self=is_train)

        labels_pred, _ = helper_knn_predict(nn_indices, self.y_train, self.n_classes, self.label_dec,
                                            self.n_neighbors)
        return labels_pred

    def predict_multiple_k(self, X, k_list, is_train=False):
        """
        Find the KNN predictions for multiple k values specified via the param `k_list`. This is done efficiently
        by querying for the maximum number of nearest neighbors once and using the results. It is assumed that the
        values in `k_list` are sorted in increasing order. This is useful while performing a search for the
        best `k` value using cross-validation.

        NOTE: The maximum value in `k_list` should be <= `self.n_neighbors`.

        :param X: numpy array with the feature vectors of shape `(N, d)`, where `N` is the number of samples
                  and `d` is the dimension.
        :param k_list: list or array of k values for which predictions are to be made. Each value should be an
                       integer >= 1 and the values should be sorted in increasing order. For example,
                       `k_list = [2, 4, 6, 8, 10]`.
        :param is_train: Set to True if prediction is being done on the same data used to train.

        :return: numpy array with the class predictions corresponding to each k value in `k_list`.
                 Has shape `(len(k_list), N)`.
        """
        if k_list[-1] > self.n_neighbors:
            raise ValueError("Invalid input: maximum value in `k_list` cannot be larger than {:d}.".
                             format(self.n_neighbors))

        # Query the maximum number of nearest neighbors from `k_list`
        nn_indices, nn_distances = self.index_knn.query(X, k=k_list[-1], exclude_self=is_train)

        if self.n_jobs == 1 or len(k_list) == 1:
            labels_pred = np.array(
                [helper_knn_predict(nn_indices, self.y_train, self.n_classes, self.label_dec, k)[0] for k in k_list],
                dtype=self.labels_dtype
            )
        else:
            helper_partial = partial(helper_knn_predict, nn_indices, self.y_train, self.n_classes, self.label_dec)
            pool_obj = multiprocessing.Pool(processes=self.n_jobs)
            outputs = []
            _ = pool_obj.map_async(helper_partial, k_list, callback=outputs.extend)
            pool_obj.close()
            pool_obj.join()

            labels_pred = np.array([tup[0] for tup in outputs], dtype=self.labels_dtype)

        return labels_pred

    def predict_proba(self, X, is_train=False):
        """
        Estimate the probability of each class along with the predicted most-frequent class.

        :param X: numpy array with the feature vectors of shape `(N, d)`, where `N` is the number of samples
                  and `d` is the dimension.
        :param is_train: Set to True if prediction is being done on the same data used to train.

        :return:
            - numpy array with the class predictions, of shape `(N, )`.
            - numpy array with the estimated probability of each class, of shape `(N, self.n_classes)`.
              Each row should sum to 1.
        """
        # Get the indices of the nearest neighbors from the training set
        nn_indices, nn_distances = self.index_knn.query(X, k=self.n_neighbors, exclude_self=is_train)

        labels_pred, counts = helper_knn_predict(nn_indices, self.y_train, self.n_classes, self.label_dec,
                                                 self.n_neighbors)
        proba = counts / self.n_neighbors

        return labels_pred, proba

    def fit_predict(self, X, y):
        """
        Fit a model and predict on the training data.

        :param X: numpy array with the feature vectors of shape `(N, d)`, where `N` is the number of samples
                  and `d` is the dimension.
        :param y: numpy array of class labels of shape `(N, )`.
        :return: numpy array with the class predictions, of shape `(N, )`.
        """
        self.fit(X, y)
        return self.predict(X, is_train=True)
