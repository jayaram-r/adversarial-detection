import numpy as np


def multinomial_estimation(data, alpha_prior=None):
    """
    Parameter estimation for a multinomial model using count data. Maximumum likelihood (ML) estimation is done if
    `alpha_prior` is not specified; else a maximum-a-posteriori (MAP) estimation is done using a Dirichlet prior
    with the hyper-parameters given by `alpha_prior`.

    Setting each value of `alpha_prior` to `0.5` corresponds to the non-informative Jeffrey's prior.
    Setting each value of `alpha_prior` to `1` corresponds to the uniform prior, which also leads to ML estimation.
    In general, values in `alpha_prior` function as a pseudo (prior) count for each category.

    :param data: numpy array with the count data. Shape `(n, m)`, where `n` is the number of samples and `m` is
                 the number of categories. Each value should be an integer and each row should sum to the same
                 value, the number of trials per sample.
    :param alpha_prior: None or a numpy array with the hyper-parameters of a Dirichlet prior distribution.
                        The array should have shape `(data.shape[1], )` with positive values. When `alpha` is None,
                        it is set to the default value of all ones, which leads to a uniform prior and maximum
                        likelihood estimation.
    :return:
        proba: numpy array with the multinomial probability estimates per category. Has shape `(m, )`, with
               values that sum to 1.
    """
    n_samp, n_cat = data.shape
    v = np.sum(data, axis=1)
    if np.any(np.abs(v - v[0]) > 0.):
        raise ValueError("All columns of the input `data` should sum to the same value, the number of trials "
                         "per sample.")

    n_trials = int(v[0])
    if alpha_prior is None:
        # Uniform prior which results in the maximum likelihood estimate
        alpha_prior = np.ones(n_cat)

    den = np.sum(alpha_prior) - n_cat + n_samp * n_trials
    proba = (alpha_prior - 1. + np.sum(data, axis=0)) / max(den, 1.0)

    return proba


def special_dirichlet_prior(n_cat, eps=0.1):
    """
    When all the alpha values are set to 1, the estimate coincides with the maximum likelihood estimate. In this
    case the alpha values sum to `m` (the number of categories).

    What if we knew that a particular category `i` is more likely (important) than the other categories? We can set
    `alpha_i` to a value close to but slightly smaller than 2, and the values of `alpha_j` (j != i) close to but
    slightly larger than 1 such that the sum of alpha values equals `m + 1`.
    One such choice that emphasizes the i-th category is given by:
    ```
    alpha_i = 1 + 1 - ((m - 1) / m) eps
    alpha_j = 1 + eps / m`, for all j != i
    ```
    With this choice, the sum of `alpha` values will be equal to `m + 1` and the i-th category takes a majority
    chunk of the additional 1 sample. This function returns an array of shape `(m, m)`, where the i-th row has the
    alpha values for the case when the i-th category is more likely than the rest.

    :param n_cat: number of categories. int value > 1.
    :param eps: a small value in (0, 1).

    :return: array where each row has the prior values as explained earlier. Has shape `(n_cat, n_cat)`.
    """
    alpha = (1 + eps / n_cat) * np.ones((n_cat, n_cat))
    v = 2. - ((n_cat - 1.) / n_cat) * eps
    np.fill_diagonal(alpha, v)

    return alpha
