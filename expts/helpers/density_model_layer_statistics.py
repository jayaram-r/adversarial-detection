"""
Probability density models for the joint distribution of the test statistics from different layers of a DNN.
The models can be made conditional on either the predicted class or the source class.

We use a multivariate log-normal mixture as the parametric density model for the test statistics because they are
usually non-negative valued. This is essentially equivalent to modeling the log of the test statistics using a
multivariate mixture of Gaussian densities. The number of mixture components is set using the bayesian information
criterion (BIC) for model complexity.

"""
import numpy as np
import logging
from sklearn.mixture import GaussianMixture
from constants import SEED_DEFAULT

logging.basicConfig(level=logging.INFO)
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
    if nd <= 50:
        # low dimensional
        if ns >= (100 * nd):
            covar_types = ['full', 'diag', 'tied']
        elif ns >= (10 * nd):
            covar_types = ['diag', 'tied']
        else:
            covar_types = ['diag', 'spherical']
    elif nd <= 250:
        # medium dimensional
        if ns >= (10 * nd):
            covar_types = ['diag', 'tied']
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
    Calculate the negative log of the probability density of each point in `data` under the model `model`.
    Points (rows) of `data` which have a low probability under the model will get a high score (more anomalous).

    Same as the function `train_log_normal_mixture`.
    :param data:
    :param model:
    :param log_transform:

    :return: numpy array of scores for each point in `data`. Has shape `(data.shape[0], )`.
    """
    if log_transform:
        # Ensure that the data has only non-negative values and return the log of the values
        data = log_transform_data(data)

    return -1.0 * model.score_samples(data)