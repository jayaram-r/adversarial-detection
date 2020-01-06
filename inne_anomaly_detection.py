"""
Isolation-based anomaly detection using nearest-neighbor ensembles (iNNE) method.

Bandaragoda, Tharindu R., et al. "Efficient anomaly detection by isolation using nearest neighbour ensemble."
2014 IEEE International Conference on Data Mining Workshop. IEEE, 2014.

Bandaragoda, Tharindu R., et al. "Isolation‐based anomaly detection using nearest‐neighbor ensembles."
Computational Intelligence 34.4 (2018): 968-998.

"""
import numpy as np


def prob_sample_inclusion(N, psi, t):
    """
    Given `N` data points, sub-sample size `psi`, and ensembles size (number of sub-samples) `t`, what is the
    probability that a data point will be included in at least one of the `t` sub-samples (i.e. selected at
    least once).

    :param N: (int) number of data points.
    :param psi: (int) sub-sample size, which should be <= N.
    :param t: (int) number of sub-samples or ensemble size.

    :return: probability value in [0, 1].
    """
    psi = min(psi, N)
    return 1. - (1. - psi / float(N)) ** t


def min_subsample_size(N, t=100, eps=0.05):
    """
    Given the number of data points `N` and the ensemble size `t`, find the minimum sub-sample size `psi` such that
    any data point has a probability `1 - eps` of being selected at least once.

    :param N: (int) number of data points.
    :param t: (int) number of sub-samples or ensemble size.
    :param eps: (float) small positive value.

    :return: minimum sub-sample size.
    """
    psi = N * (1. - np.exp(np.log(eps) / t))
    return int(np.ceil(psi))


def min_ensemble_size(N, psi, eps=0.05):
    """
    Given the number of data points `N` and the sub-sample size `psi`, find the minimum ensemble size `t` such
    that any data point has a probability `1 - eps` of being selected at least once.

    :param N: (int) number of data points.
    :param psi: (int) sub-sample size, which should be <= N.
    :param eps: (float) small positive value.

    :return: minimum ensemble size.
    """
    if psi < N:
        t = np.log(eps) / np.log(1. - psi / float(N))
        return int(np.ceil(t))
    else:
        return 1
