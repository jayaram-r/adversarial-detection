"""
Probability density models for the joint distribution of the test statistics from different layers of a DNN.
The models can be made conditional on either the predicted class or the source class.

We use a multivariate log-normal mixture as the parametric density model for the test statistics because they are
usually non-negative valued. This is essentially equivalent to modeling the log of the test statistics using a
multivariate mixture of Gaussian densities. The number of mixture components is set using the bayesian information
criterion (BIC) for model complexity.

"""
import numpy as np
import sys
import logging
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal, chi2
from  helpers.utils import log_sum_exp
from helpers.constants import SEED_DEFAULT

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def log_transform_data(data_in):
    """
    Log transform the data array while suitably handling 0 and negative values.

    :param data_in: numpy data array.
    :return: log-transformed numpy data array.
    """
    mask = data_in < 0.
    if np.any(mask):
        raise ValueError("Data array has negative values. Cannot proceed.")

    # Replacing any 0 values with a very small positive value
    v = np.min(data_in[data_in > 0.])
    v = min(np.log10(v) - 1, -16)

    return np.log(np.clip(data_in, 10 ** v, None))


def select_covar_types(nd, ns):
    """
    Heuristics to choose the types of covariance matrix to explore based on the data dimension and the (effective)
    number of samples per mixture component.

    :param nd: data dimension.
    :param ns: number of samples.
    :return: list of covariance types.
    """
    if nd <= 20:
        # low dimensional
        if ns >= (10 * nd):
            covar_types = ['full', 'tied']
        else:
            covar_types = ['tied', 'diag']
    elif nd <= 250:
        # medium dimensional
        if ns >= (10 * nd):
            covar_types = ['tied', 'diag']
            if nd <= 50:
                covar_types.append('full')

        else:
            covar_types = ['diag', 'spherical']
    else:
        # high dimensional
        covar_types = ['diag', 'spherical']

    return covar_types


def train_log_normal_mixture(data,
                             log_transform=True,
                             min_n_components=1,
                             max_n_components=30,
                             step_n_components=1,
                             covar_types=None,
                             n_init=10,
                             max_iter=500,
                             num_successive_steps=3,
                             seed_rng=SEED_DEFAULT):
    """
    Fit a log-normal mixture density model to the data by searching over the number of mixture components and
    exploring suitable covariance types. Select the best number of components and covariance type using the
    bayesian information criterion (BIC) for model selection.

    :param data: numpy data array of shape `(ns, nd)`, where `ns` is the number of samples and `nd` is the number
                 of dimensions (features).
    :param log_transform: Set to True in order to log-transform the data.
    :param min_n_components: int value specifying the lower end of the search range for the number of components.
    :param max_n_components: int value specifying the upper end of the search range for the number of components.
    :param step_n_components: int value specifying the step value of the search range for the number of components.
    :param covar_types: None or a list of covariance types to explore. If set to `None`, this is decided
                        automatically based on the data dimension and number of samples. Valid types include:
                        'full', 'tied', 'diag', 'spherical'.
    :param n_init: int value specifying the number of random initializations used for the EM algorithm.
    :param max_iter: int value specifying the max number of iterations of the EM algorithm.
    :param num_successive_steps: int value specifying the number of succesive steps of BIC increase that leads
                                 to stop increasing the number of mixture components. This will avoid searching over
                                 the entire range of number of components when it is evident that the increasing
                                 model complexity is not supported by the data.
    :param seed_rng: seed value for the random number generator.

    :return: model instance of the class `GaussianMixture` that was found to be the best fit to the data.
    """
    ns, nd = data.shape
    if log_transform:
        # Ensure that the data has only non-negative values and return the log of its values
        data = log_transform_data(data)

    covar_types_orig = covar_types
    range_n_components = np.arange(min_n_components, max_n_components + 1, step_n_components)

    bic_min = np.infty
    mod_best = None
    cnt = 0
    for k in range_n_components:
        if not covar_types_orig:
            # Effective number of samples per mixture component
            ns_eff = int(np.round(float(ns) / k))
            covar_types = select_covar_types(nd, ns_eff)
        else:
            covar_types = covar_types_orig

        mod_best_curr = None
        bic_min_curr = np.infty
        for ct in covar_types:
            mod_gmm = GaussianMixture(n_components=k, covariance_type=ct, max_iter=max_iter, n_init=n_init,
                                      random_state=seed_rng, verbose=0)
            _ = mod_gmm.fit(data)
            v = mod_gmm.bic(data)
            logger.info(" #components = {:d}, covariance type = {}, BIC score = {:.4f}".format(k, ct, v))
            if v < bic_min_curr:
                bic_min_curr = v
                mod_best_curr = mod_gmm

        if bic_min_curr < bic_min:
            bic_min = bic_min_curr
            mod_best = mod_best_curr
            cnt = 0
        else:
            # BIC increasing
            cnt += 1

        if cnt >= num_successive_steps:
            break

    logger.info(" Model training complete.")
    logger.info(" Best model: #components = {:d}, covariance type = {}, BIC score = {:.4f}".
                format(mod_best.n_components, mod_best.covariance_type, bic_min))
    return mod_best


def score_log_normal_mixture(data, model, log_transform=True):
    """
    Calculate the log of the probability density of each point in `data` under the Gaussian mixture model `model`.
    Same as the function `train_log_normal_mixture`.
    :param data:
    :param model:
    :param log_transform:

    :return: numpy array with the log-likelihood values of each point (row) in `data` under the given model.
             Has shape `(data.shape[0], )`.
    """
    if log_transform:
        # Ensure that the data has only non-negative values and return the log of the values
        data = log_transform_data(data)

    return model.score_samples(data)


def log_pvalue_gmm_approx(data, model, log_transform=True):
    """
    Log of the p-value of a set of points in `data` relative to a Gaussian mixture model.
    This is an approximation to the p-value.

    :param data: Numpy array of shape `(n, d)`, where `n` is the number of points and `d` is the dimension.
    :param model: Trained Gaussian mixture model object.
    :param log_transform: Set to True in order to log-transform the data prior to analysis.

    :return: numpy array of shape `(n, )` with the log of the p-values for each point in `data`.
    """
    if log_transform:
        data = log_transform_data(data)

    # number of samples `n` and the number of dimensions `d`
    n, d = data.shape
    # number of components
    k = model.n_components
    # Component posterior probabilities; shape (n, k)
    post_prob = model.predict_proba(data)

    chi2_cdf = np.zeros((n, k))
    for j in range(k):
        # component j
        mu = model.means_[j, :]
        if model.covariance_type == 'full':
            # has shape (k, d, d)
            cov = model.covariances_[j, :, :]
        elif model.covariance_type == 'tied':
            # has shape (d, d)
            cov = model.covariances_
        elif model.covariance_type == 'diag':
            # has shape (k, d)
            cov = model.covariances_[j, :]
        else:
            # has shape (k, )
            cov = model.covariances_[j]

        # Mahalanobis distance of the points `data` from the mean of component `j` can be calculated from the
        # log probability density
        dens = multivariate_normal(mean=mu, cov=cov)
        dist_mahal = -2. * (dens.logpdf(data) - dens.logpdf(mu))

        # CDF of the Chi-squared distribution (`d` degrees of freedom) evaluated at the mahalanobis distance values
        chi2_cdf[:, j] = chi2.cdf(dist_mahal, d)

    tmp_arr = 1. - np.sum(post_prob * chi2_cdf, axis=1)
    return np.log(np.clip(tmp_arr, sys.float_info.min, None))
