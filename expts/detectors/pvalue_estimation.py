# Functions for empirical p-value estimation
import numpy as np
from numba import njit, prange
from helpers.constants import NUM_BOOTSTRAP


@njit(parallel=True)
def pvalue_score(scores_null, scores_obs, log_transform=False, bootstrap=True, n_bootstrap=NUM_BOOTSTRAP):
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
    n_samp = scores_null.shape[0]
    n_obs = scores_obs.shape[0]
    p = np.zeros(n_obs)
    for i in prange(n_obs):
        for j in range(n_samp):
            if scores_null[j] >= scores_obs[i]:
                p[i] += 1.

        p[i] = p[i] / n_samp

    if bootstrap:
        ind_null_repl = np.random.choice(np.arange(n_samp), size=(n_bootstrap, n_samp), replace=True)
        p_sum = p
        for b in prange(n_bootstrap):
            p_curr = np.zeros(n_obs)
            for i in range(n_obs):
                for j in ind_null_repl[b, :]:
                    if scores_null[j] >= scores_obs[i]:
                        p_curr[i] += 1.

                p_curr[i] = p_curr[i] / n_samp

            p_sum += p_curr

        # Average p-value from the bootstrap replications
        p = p_sum / (n_bootstrap + 1.)

    p[p < eps] = eps
    if log_transform:
        return -np.log(p)
    else:
        return p


@njit(parallel=True)
def pvalue_score_bivar(scores_null, scores_obs, log_transform=False, bootstrap=True, n_bootstrap=NUM_BOOTSTRAP):
    """
    Calculate the empirical p-values of the bivariate observed scores `scores_obs` with respect to the scores from
    the null distribution `scores_null`. Bootstrap resampling can be used to get better estimates of the p-values.

    :param scores_null: numpy array of shape `(m, 2)` with the scores from the null distribution.
    :param scores_obs: numpy array of shape `(n, 2)` with the observed scores.
    :param log_transform: set to True to apply negative log transform to the p-values.
    :param bootstrap: Set to True to calculate a bootstrap resampled estimate of the p-value.
    :param n_bootstrap: number of bootstrap resamples to use.
    :param n_jobs: number of jobs to execute in parallel.

    :return: numpy array with the p-values or negative-log-transformed p-values. Has shape `(scores_obs.shape[0], )`.
    """
    eps = 1e-16
    n_samp = scores_null.shape[0]
    n_obs = scores_obs.shape[0]
    p = np.zeros(n_obs)
    for i in prange(n_obs):
        for j in range(n_samp):
            if (scores_null[j, 0] >= scores_obs[i, 0]) and (scores_null[j, 1] >= scores_obs[i, 1]):
                p[i] += 1.

        p[i] = p[i] / n_samp

    if bootstrap:
        ind_null_repl = np.random.choice(np.arange(n_samp), size=(n_bootstrap, n_samp), replace=True)
        p_sum = p
        for b in prange(n_bootstrap):
            p_curr = np.zeros(n_obs)
            for i in range(n_obs):
                for j in ind_null_repl[b, :]:
                    if (scores_null[j, 0] >= scores_obs[i, 0]) and (scores_null[j, 1] >= scores_obs[i, 1]):
                        p_curr[i] += 1.

                p_curr[i] = p_curr[i] / n_samp

            p_sum += p_curr

        # Average p-value from the bootstrap replications
        p = p_sum / (n_bootstrap + 1.)

    p[p < eps] = eps
    if log_transform:
        return -np.log(p)
    else:
        return p


def pvalue_score_all_pairs(scores_null, scores_obs, log_transform=False, bootstrap=True, n_bootstrap=NUM_BOOTSTRAP):
    n_obs, n_feat = scores_obs.shape
    n_pairs = int(0.5 * n_feat * (n_feat - 1))
    p_vals = np.zeros((n_obs, n_pairs))
    k = 0
    for i in range(n_feat - 1):
        for j in range(i + 1, n_feat):
            p_vals[:, k] = pvalue_score_bivar(
                scores_null[:, [i, j]], scores_obs[:, [i, j]], log_transform=log_transform,
                bootstrap=bootstrap, n_bootstrap=n_bootstrap
            )
            k += 1

    return p_vals
