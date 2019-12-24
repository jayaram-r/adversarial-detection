"""
Some custom distance metrics and similarity measures.
"""
import numpy as np
from numba import njit, float64, int64
from numba.types import Tuple


@njit(fastmath=True)
def distance_norm_3tensors(x, y, shape=(1, 1, 1), norm_type=(2, 2, 2)):
    """
    Distance between two 3rd order (rank 3) tensors under the specified type of norm.
    The tensors `x` and `y` should be flattened into 1D numpy arrays before calling this function.
    If `xt` and `yt` are the tensors, each of shape `shape`, they can be flattened into a 1D array using
    `x = xt.reshape(-1)` (and likewise for `yt`). The shape is passed as input, which is used by the function
    to reshape the arrays to tensors.

    The inputs are taken as 1d arrays in order to be consistent  with the function signature required by
    another library.

    The input `norm_type` is a tuple of length three that takes the norm parameters `p, q, r`, which together specify
    the type of norm used. They should be integers >= 1.
    For example, the default value `norm_type=(2, 2, 2)` computes the `l2` or Euclidean norm of the flattened tensor.
    The value `norm_type=(1, 1, 1)` computes the `l1` norm of the flattened tensor.

    The special case `p = -1` with finite `q` and `r` calculates the norm with `p = \infty`. The value `-1` is
    used instead of `np.infty` because numba does not handle tuple with mixed float and int values.

    :param x: numpy array of shape `(n, )` with the first flattened tensor.
    :param y: numpy array of shape `(n, )` with the second flattened tensor.
    :param shape: tuple of three values specifying the shape of the tensors. This is a required argument.
                  The default value `(1, 1, 1)` is a dummy value used to specify the type.
    :param norm_type: tuple of three values `(p, q, r)` that together define the type of norm to be used.
                      `q` and `r` should be integers >= 1. `p = -1` is a special value that calculates the
                      norm with `p = \infty`; otherwise `p` should be an integer >= 1.

    :return: distance value which is a non-negative float.
    """
    zt = np.abs(x - y).reshape(shape)
    # q / r
    pow1 = norm_type[1] / norm_type[2]
    if norm_type[0] > 0:
        # p / q
        pow2 = norm_type[0] / norm_type[1]
        s = 0.
        for i in range(shape[0]):
            sj = 0.
            for j in range(shape[1]):
                sj += (np.sum(zt[i, j, :] ** norm_type[2]) ** pow1)

            s += (sj ** pow2)

        # power 1 / p
        dist = s ** (1. / norm_type[0])
    else:
        # p = \infty
        s = -1.0
        for i in range(shape[0]):
            sj = 0.
            for j in range(shape[1]):
                sj += (np.sum(zt[i, j, :] ** norm_type[2]) ** pow1)

            if sj > s:
                s = sj

        dist = s ** (1. / norm_type[1])

    return dist


@njit(fastmath=True)
def distance_angular_3tensors(x, y, shape=(1, 1, 1)):
    """
    Cosine angular distance between two 3rd order (rank 3) tensors.
    The tensors `x` and `y` should be flattened into 1D numpy arrays before calling this function.
    If `xt` and `yt` are the tensors, each of shape `shape`, they can be flattened into a 1D array using
    `x = xt.reshape(-1)` (and likewise for `yt`). The shape is passed as input, which is used by the function
    to reshape the arrays to tensors.

    The inputs are taken as 1d arrays in order to be consistent  with the function signature required by
    another library.

    :param x: numpy array of shape `(n, )` with the first flattened tensor.
    :param y: numpy array of shape `(n, )` with the second flattened tensor.
    :param shape: tuple of three values specifying the shape of the tensors. This is a required argument.
                  The default value `(1, 1, 1)` is a dummy value used to specify the type.
    :return: distance value which should be in the range [0, \pi].
    """
    xt = x.reshape(shape)
    yt = y.reshape(shape)
    s = 0.
    for i in range(shape[0]):
        val1 = np.sum(xt[i, :, :] * yt[i, :, :])
        val2 = np.sum(xt[i, :, :] * xt[i, :, :]) ** 0.5
        val3 = np.sum(yt[i, :, :] * yt[i, :, :]) ** 0.5
        if val2 > 0. and val3 > 0.:
            s += (val1 / (val2 * val3))
        elif val2 <= 0. and val3 <= 0.:
            # Both vector of 0s
            s += 1.

    # Angular distance is the cosine-inverse of the average cosine similarity, divided by `pi` to normalize
    # the distance to the range `[0, 1]`
    s = max(-1., min(1., s / shape[0]))

    return np.arccos(s)


@njit()
def distance_SNN(x, y):
    """
    Shared nearest neighbor distance metric. This is a secondary (ranking-based) distance measure.
    First, a standard distance metric is used to find the K nearest neighbors (K-NN) of each point from a fixed set
    of `N` points. If `N` and/or the dimensionality is large, this can be done by constructing an approximate
    nearest neighbor index and querying it.

    The shared nearest neighbor (SNN) similarity between two points `x` and `y` is defined as the size of the
    intersection of their K-NN sets, divided by `K`. This is equal to the cosine similarity between the binary
    representation of the two points (of length `N`) in which the presence of as neighbor is indicated by 1
    and absence by 0. This is easy to see because their inner product will be the number of overlapping neighbors,
    and the norm of each vector will be `sqrt(K)`.

    The cosine similarity is translated into a distance by taking the inverse-cosine which gives and angle
    value between `[0, \pi]`.

    For justification and advantages of this distance metric, see this paper:
    Houle, Michael E., et al. "Can shared-neighbor distances defeat the curse of dimensionality?."
    International Conference on Scientific and Statistical Database Management. Springer, Berlin, Heidelberg, 2010.

    :param x: numpy array of shape `(k1, )` and dtype `int` with the indices of the neighbors.
    :param y: numpy array of shape `(k2, )` and dtype `int` with the indices of the neighbors.
    :return: SNN distance which should be in the range [0, \pi].
    """
    # Neighborhood size of each point
    s_x = x.shape[0]
    s_y = y.shape[0]

    # Size of neighborhood overlap.
    # Using loops since the numpy functions such as `numpy.isin` and `numpy.intersect1d` are not supported by numba
    s_xy = 0.
    for i in x:
        for j in y:
            if i == j:
                s_xy += 1.
                break

    cs = s_xy / ((s_x * s_y) ** 0.5)
    # Clip values to the range `[-1, 1]`, the domain of arc-cosine
    dist = np.arccos(max(-1., min(1., cs)))

    return dist


@njit(Tuple((int64[:, :], float64[:, :]))(int64[:, :], float64[:, :]))
def remove_self_neighbors(index_neighbors_, distance_neighbors_):
    """
    Given the index and distances of k nearest neighbors of a list of query points, remove points from their
    own neighbor list.

    :param index_neighbors_: numpy array of the index of `k` neighbors for a list of points. Has shape `(n, k)`,
                             where `n` is the number of query points.
    :param distance_neighbors_: numpy array of the distance of `k` neighbors for a list of points.
                                Also has shape `(n, k)`.

    :return: (index_neighbors, distance_neighbors), where each of them has shape `(n, k - 1)`.
    """
    n, k = index_neighbors_.shape
    index_neighbors = np.zeros((n, k - 1), dtype=index_neighbors_.dtype)
    distance_neighbors = np.zeros((n, k - 1), dtype=distance_neighbors_.dtype)
    for i in range(n):
        j1 = j2 = 0
        while j1 < (k - 1) and j2 < k:
            if index_neighbors_[i, j2] != i:
                index_neighbors[i, j1] = index_neighbors_[i, j2]
                distance_neighbors[i, j1] = distance_neighbors_[i, j2]
                j1 += 1

            j2 += 1

    return index_neighbors, distance_neighbors
