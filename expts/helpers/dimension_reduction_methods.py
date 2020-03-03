"""
Implementation of the following dimensionality reduction methods, which learn locality (neighborhood) preserving
linear projections that are defined everywhere, i.e. they can be applied to new test data unlike methods like
locally linear embedding (LLE), Laplacian eigenmaps, and IsoMap.

- Locality preserving projections (LPP)
1. He, Xiaofei, and Partha Niyogi. "Locality preserving projections." Advances in neural information processing
systems. 2004.
2. He, Xiaofei, et al. "Face recognition using LaplacianFaces." IEEE Transactions on Pattern Analysis & Machine
Intelligence 3 (2005): 328-340.

 - Orthogonal neighborhood preserving projections (ONPP)
3. Kokiopoulou, Effrosyni, and Yousef Saad. "Orthogonal neighborhood preserving projections: A projection-based
dimensionality reduction technique." IEEE Transactions on Pattern Analysis and Machine Intelligence
29.12 (2007): 2143-2156.

- Orthogonal locality preserving projections (OLPP)
This method is based on a slight modification of the LPP formulation. The projection matrix is constrained to
be orthonormal. This method is described in [3] and the results show that the orthogonal projections found by
ONPP and OLPP have the best performance.

Note that [4] also implements a variation of the OLPP method, but we do not implement their method here.
4. Cai, Deng, et al. "Orthogonal LaplacianFaces for face recognition." IEEE transactions on image processing
15.11 (2006): 3608-3614.

"""
import numpy as np
import sys
from scipy import sparse
from scipy.linalg import eigh, eigvalsh, solve
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import multiprocessing
from functools import partial
from helpers.lid_estimators import estimate_intrinsic_dimension
from helpers.knn_index import KNNIndex
from helpers.utils import get_num_jobs
from helpers.constants import (
    NEIGHBORHOOD_CONST,
    SEED_DEFAULT,
    METRIC_DEF
)
import logging
try:
    import cPickle as pickle
except:
    import pickle


logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

METHODS_LIST = ['LPP', 'OLPP', 'NPP', 'ONPP', 'PCA']


def helper_distance(data, nn_indices, metric, metric_kwargs, i):
    """
    Helper function to calculate pairwise distances.
    """
    if metric_kwargs is None:
        pd = pairwise_distances(
            data[i, :].reshape(1, -1),
            Y=data[nn_indices[i, :], :],
            metric=metric,
            n_jobs=1
        )
    else:
        pd = pairwise_distances(
            data[i, :].reshape(1, -1),
            Y=data[nn_indices[i, :], :],
            metric=metric,
            n_jobs=1,
            **metric_kwargs
        )
    return pd[0, :]


def calculate_heat_kernel(data, nn_indices, heat_kernel_param, metric, metric_kwargs=None, n_jobs=1):
    """
    Calculate the heat kernel values for sample pairs in `data` that are nearest neighbors (given by `nn_indices`).

    :param data: data matrix of shape `(N, d)` where `N` is the number of samples and `d` is the number of
                 dimensions.
    :param nn_indices: numpy array of shape `(N, K)` with the index of `K` nearest neighbors for each of the
                       `N` samples.
    :param heat_kernel_param: None or a float value specifying the heat kernel scale parameter. If set to None,
                              this parameter will be set to suitable value based on the median of the pairwise
                              distances.
    :param metric: distance metric string or callable.
    :param metric_kwargs: None or a dict of keyword arguments to be passed to the distance metric.
    :param n_jobs: number of CPU cores to use for parallel processing.

    :return: Heat kernel values returned as a numpy array of shape `(N, K)`.
    """
    N, K = nn_indices.shape
    if n_jobs == 1:
        dist_mat = [helper_distance(data, nn_indices, metric, metric_kwargs, i) for i in range(N)]
    else:
        helper_distance_partial = partial(helper_distance, data, nn_indices, metric, metric_kwargs)
        pool_obj = multiprocessing.Pool(processes=n_jobs)
        dist_mat = []
        _ = pool_obj.map_async(helper_distance_partial, range(N), callback=dist_mat.extend)
        pool_obj.close()
        pool_obj.join()

    dist_mat = np.array(dist_mat) ** 2
    if heat_kernel_param is None:
        # Heuristic choice: setting the kernel parameter such that the kernel value is equal to `0.1` for the
        # maximum pairwise distance among all neighboring pairs
        heat_kernel_param = np.percentile(dist_mat, 99.9) / np.log(10.)

    return np.exp((-1.0 / heat_kernel_param) * dist_mat)


def pca_wrapper(data, n_comp=None, cutoff=1.0, seed_rng=SEED_DEFAULT):
    """
    Find the PCA transformation for the provided data, which is assumed to be centered.

    :param data: data matrix of shape `(N, d)` where `N` is the number of samples and `d` is the number of
                 dimensions.
    :param n_comp: None or int value (>= 1) specifying the dimension (number of components) of the PCA projection.
                   If this value is specified, the variance cutoff threshold is not used.
    :param cutoff: variance cutoff value in (0, 1]. This value is used to select the number of components only if
                   `n_comp` is not specified.
    :param seed_rng: seed for random number generator.

    :return: (data_trans, mean_data, transform_pca), where
        - data_trans: Transformed, dimension reduced data matrix of shape `(N, d_red)`.
        - mean_data: numpy array with the sample mean value of each feature.
        - transform_pca: numpy array with the PCA transformation matrix.
    """
    N, d = data.shape
    mod_pca = PCA(n_components=min(N, d), random_state=seed_rng)
    _ = mod_pca.fit(data)

    # Number of components with non-zero singular values
    sig = mod_pca.singular_values_
    n1 = sig[sig > 1e-16].shape[0]
    logger.info("Number of nonzero singular values in the data matrix = {:d}".format(n1))

    # Number of components accounting for the specified fraction of the cumulative data variance
    var_cum = np.cumsum(mod_pca.explained_variance_)
    var_cum_frac = var_cum / var_cum[-1]
    ind = np.where(var_cum_frac >= cutoff)[0]
    if ind.shape[0]:
        n2 = ind[0] + 1
    else:
        n2 = var_cum.shape[0]

    logger.info("Number of principal components accounting for {:.1f} percent of the data variance = {:d}".
                format(100 * cutoff, n2))
    if n_comp is None:
        n_comp = min(n1, n2)
    else:
        n_comp = min(n1, n2, n_comp)

    logger.info("Dimension of the PCA transformed data = {:d}".format(n_comp))
    transform_pca = mod_pca.components_[:n_comp, :].T
    data_trans = np.dot(data - mod_pca.mean_, transform_pca)

    return data_trans, mod_pca.mean_, transform_pca


class LocalityPreservingProjection:
    """
    Locality preserving projection (LPP) method for dimensionality reduction [1, 2].
    Orthogonal LPP (OLPP) method based on [3].

    1. He, Xiaofei, and Partha Niyogi. "Locality preserving projections." Advances in neural information processing
    systems. 2004.
    2. He, Xiaofei, et al. "Face recognition using LaplacianFaces." IEEE Transactions on Pattern Analysis & Machine
    Intelligence 3 (2005): 328-340.
    3. Kokiopoulou, Effrosyni, and Yousef Saad. "Orthogonal neighborhood preserving projections: A projection-based
    dimensionality reduction technique." IEEE Transactions on Pattern Analysis and Machine Intelligence,
    29.12 (2007): 2143-2156.

    """
    def __init__(self,
                 dim_projection='auto',                         # 'auto' or positive integer
                 orthogonal=False,                              # True to enable Orthogonal LPP (OLPP)
                 pca_cutoff=1.0,
                 neighborhood_constant=NEIGHBORHOOD_CONST, n_neighbors=None,   # Specify one of them. If `n_neighbors`
                 # is specified, `neighborhood_constant` will be ignored.
                 shared_nearest_neighbors=False,
                 edge_weights='SNN',                            # Choices are {'simple', 'SNN', 'heat_kernel'}
                 heat_kernel_param=None,                        # Used only if `edge_weights = 'heat_kernel'`
                 metric=METRIC_DEF, metric_kwargs=None,        # distance metric and its parameter dict (if any)
                 approx_nearest_neighbors=True,
                 n_jobs=1,
                 seed_rng=SEED_DEFAULT):
        """
        :param dim_projection: Dimension of data in the projected feature space. If set to 'auto', a suitable reduced
                               dimension will be chosen by estimating the intrinsic dimension of the data. If an
                               integer value is specified, it should be in the range `[1, dim - 1]`, where `dim`
                               is the observed dimension of the data.
        :param orthogonal: Set to True to select the OLPP method. It was shown to have better performance than LPP
                           in [3].
        :param pca_cutoff: float value in (0, 1] specifying the proportion of cumulative data variance to preserve
                           in the projected dimension-reduced data. PCA is applied as a first-level dimension
                           reduction to handle potential data matrix singularity also. Set `pca_cutoff = 1.0` in
                           order to handle only the data matrix singularity.
        :param neighborhood_constant: float value in (0, 1), that specifies the number of nearest neighbors as a
                                      function of the number of samples (data size). If `N` is the number of samples,
                                      then the number of neighbors is set to `N^neighborhood_constant`. It is
                                      recommended to set this value in the range 0.4 to 0.5.
        :param n_neighbors: None or int value specifying the number of nearest neighbors. If this value is specified,
                            the `neighborhood_constant` is ignored. It is sufficient to specify either
                            `neighborhood_constant` or `n_neighbors`.
        :param shared_nearest_neighbors: Set to True in order to use the shared nearest neighbor (SNN) distance to
                                         find the K nearest neighbors. This is a secondary distance metric that is
                                         found to be better suited to high dimensional data. This will be set to
                                         True if `edge_weights = 'SNN'`.
        :param edge_weights: Weighting method to use for the edge weights. Valid choices are {'simple', 'SNN',
                             'heat_kernel'}. They are described below:
                             - 'simple': the edge weight is set to one for every sample pair in the neighborhood.
                             - 'SNN': the shared nearest neighbors (SNN) similarity score between two samples is used
                             as the edge weight. This will be a value in [0, 1].
                             - 'heat_kernel': the heat (Gaussian) kernel with a suitable scale parameter defines the
                             edge weight.
        :param heat_kernel_param: Heat kernel scale parameter. If set to `None`, this parameter is set automatically
                                  based on the median of the pairwise distances between samples. Else a positive
                                  real value can be specified.
        :param metric: string or a callable that specifies the distance metric to be used for the SNN similarity
                       calculation. This is used only if `edge_weights = 'SNN'`.
        :param metric_kwargs: optional keyword arguments required by the distance metric specified in the form of a
                              dictionary. Again, this is used only if `edge_weights = 'SNN'`.
        :param approx_nearest_neighbors: Set to True in order to use an approximate nearest neighbor algorithm to
                                         find the nearest neighbors. This is recommended when the number of points is
                                         large and/or when the dimension of the data is high.
        :param n_jobs: Number of parallel jobs or processes. Set to -1 to use all the available cpu cores.
        :param seed_rng: int value specifying the seed for the random number generator.
        """
        self.dim_projection = dim_projection
        self.orthogonal = orthogonal
        self.pca_cutoff = pca_cutoff
        self.neighborhood_constant = neighborhood_constant
        self.n_neighbors = n_neighbors
        self.shared_nearest_neighbors = shared_nearest_neighbors
        self.edge_weights = edge_weights.lower()
        self.heat_kernel_param = heat_kernel_param
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.approx_nearest_neighbors = approx_nearest_neighbors
        self.n_jobs = get_num_jobs(n_jobs)
        self.seed_rng = seed_rng

        if self.edge_weights not in {'simple', 'snn', 'heat_kernel'}:
            raise ValueError("Invalid value '{}' for parameter 'edge_weights'".format(self.edge_weights))

        if self.edge_weights == 'snn':
            self.shared_nearest_neighbors = True

        self.mean_data = None
        self.index_knn = None
        self.adjacency_matrix = None
        self.incidence_matrix = None
        self.laplacian_matrix = None
        self.transform_pca = None
        self.transform_lpp = None
        self.transform_comb = None

    def fit(self, data):
        """
        Find the optimal projection matrix for the given data points.

        :param data: data matrix of shape `(N, d)` where `N` is the number of samples and `d` is the number of
                     dimensions.
        :return:
        """
        N, d = data.shape
        if self.n_neighbors is None:
            # Set number of nearest neighbors based on the data size and the neighborhood constant
            self.n_neighbors = int(np.ceil(N ** self.neighborhood_constant))

        logger.info("Applying PCA as first-level dimension reduction step")
        data, self.mean_data, self.transform_pca = pca_wrapper(data, cutoff=self.pca_cutoff,
                                                               seed_rng=self.seed_rng)
        if self.dim_projection == 'auto':
            # Estimate the intrinsic dimension of the data and use that as the projected dimension
            id = estimate_intrinsic_dimension(data,
                                              method='two_nn',
                                              n_neighbors=self.n_neighbors,
                                              approx_nearest_neighbors=self.approx_nearest_neighbors,
                                              n_jobs=self.n_jobs,
                                              seed_rng=self.seed_rng)
            self.dim_projection = int(np.ceil(id))
            logger.info("Estimated intrinsic dimension of the (PCA-projected) data = {:.2f}.".format(id))

        if self.dim_projection >= data.shape[1]:
            self.dim_projection = data.shape[1]

        logger.info("Dimension of the projected subspace = {:d}".format(self.dim_projection))

        # Create a KNN index for all nearest neighbor tasks
        self.index_knn = KNNIndex(
            data, n_neighbors=self.n_neighbors,
            metric=self.metric, metric_kwargs=self.metric_kwargs,
            shared_nearest_neighbors=self.shared_nearest_neighbors,
            approx_nearest_neighbors=self.approx_nearest_neighbors,
            n_jobs=self.n_jobs,
            seed_rng=self.seed_rng
        )

        # Create the symmetric adjacency matrix, diagonal incidence matrix, and the graph Laplacian matrix
        # for the data points
        self.create_laplacian_matrix(data)

        # Solve the generalized eigenvalue problem and take the eigenvectors corresponding to the smallest
        # eigenvalues as the columns of the projection matrix
        logger.info("Solving the generalized eigenvalue problem to find the optimal projection matrix.")
        data_trans = data.T
        # X^T L X
        lmat = sparse.csr_matrix.dot(data_trans, self.laplacian_matrix).dot(data)
        if self.orthogonal:
            # Orthogonal LPP
            eig_values, eig_vectors = eigh(lmat, eigvals=(0, self.dim_projection - 1))
        else:
            # Standard LPP
            # X^T D X
            rmat = sparse.csr_matrix.dot(data_trans, self.incidence_matrix).dot(data)
            eig_values, eig_vectors = eigh(lmat, b=rmat, eigvals=(0, self.dim_projection - 1))

        # `eig_vectors` is a numpy array with each eigenvector along a column. The eigenvectors are ordered
        # according to increasing eigenvalues.
        # `eig_vectors` will have shape `(data.shape[1], self.dim_projection)`
        self.transform_lpp = eig_vectors

        self.transform_comb = np.dot(self.transform_pca, self.transform_lpp)

    def transform(self, data, dim=None):
        """
        Transform the given data by first subtracting the mean and then applying the linear projection.
        Optionally, you can specify the dimension of the transformed data using `dim`. This cannot be larger
        than `self.dim_projection`.

        :param data: numpy array of shape `(N, d)` with `N` samples in `d` dimensions.
        :param dim: If set to `None`, the dimension of the transformed data is `self.dim_projection`.
                    Else `dim` can be set to a value <= `self.dim_projection`. Doing this basically takes only
                    the `dim` top eigenvectors.

        :return:
            - data_trans: numpy array of shape `(N, dim)` with the transformed, dimension-reduced data.
        """
        if dim is None:
            data_trans = np.dot(data - self.mean_data, self.transform_comb)
        else:
            data_trans = np.dot(data - self.mean_data, self.transform_comb[:, 0:dim])

        return data_trans

    def fit_transform(self, data, dim=None):
        """
        Fit the model and transform the given data.

        :param data: numpy array of shape `(N, d)` with `N` samples in `d` dimensions.
        :param dim: same as the `transform` method.

        :return:
            - data_trans: numpy array of shape `(N, dim)` with the transformed, dimension-reduced data.
        """
        self.fit(data)
        return self.transform(data, dim=dim)

    def create_laplacian_matrix(self, data):
        """
        Calculate the graph Laplacian matrix for the given data.

        :param data: data matrix of shape `(N, d)` where `N` is the number of samples and `d` is the number of
                     dimensions.
        :return:
        """
        # Find the `self.n_neighbors` nearest neighbors of each point
        nn_indices, nn_distances = self.index_knn.query_self(k=self.n_neighbors)

        N, K = nn_indices.shape
        row_ind = np.array([[i] * K for i in range(N)], dtype=np.int).ravel()
        col_ind = nn_indices.ravel()
        if self.edge_weights == 'simple':
            vals = np.ones(N * K)
        elif self.edge_weights == 'snn':
            # SNN distance is the cosine-inverse of the SNN similarity. The range of SNN distances will
            # be [0, pi / 2]. Hence, the SNN similarity will be in the range [0, 1].
            vals = np.clip(np.cos(nn_distances).ravel(), 0., None)
        else:
            # Heat kernel
            vals = calculate_heat_kernel(
                data, nn_indices, self.heat_kernel_param, self.metric, metric_kwargs=self.metric_kwargs,
                n_jobs=self.n_jobs
            ).ravel()

        # Adjacency or edge weight matrix (W)
        mat_tmp = sparse.csr_matrix((vals, (row_ind, col_ind)), shape=(N, N))
        self.adjacency_matrix = 0.5 * (mat_tmp + mat_tmp.transpose())

        # Incidence matrix (D)
        vals_diag = self.adjacency_matrix.sum(axis=1)
        vals_diag = np.array(vals_diag[:, 0]).ravel()
        ind = np.arange(N)
        self.incidence_matrix = sparse.csr_matrix((vals_diag, (ind, ind)), shape=(N, N))

        # Graph laplacian matrix (L = D - W)
        self.laplacian_matrix = self.incidence_matrix - self.adjacency_matrix


class NeighborhoodPreservingProjection:
    """
    Neighborhood preserving projection (NPP) method for dimensionality reduction. Also known as neighborhood
    preserving embedding (NPE) [1].
    Orthogonal neighborhood preserving projection (ONPP) method is based on [2].

    1. He, Xiaofei, et al. "Neighborhood preserving embedding." Tenth IEEE International Conference on Computer
       Vision (ICCV'05) Volume 1. Vol. 2. IEEE, 2005.
    2. Kokiopoulou, Effrosyni, and Yousef Saad. "Orthogonal neighborhood preserving projections: A projection-based
    dimensionality reduction technique." IEEE Transactions on Pattern Analysis and Machine Intelligence,
    29.12 (2007): 2143-2156.

    """
    def __init__(self,
                 dim_projection='auto',                         # 'auto' or positive integer
                 orthogonal=False,                              # True to enable Orthogonal NPP (ONPP) method
                 pca_cutoff=1.0,
                 neighborhood_constant=NEIGHBORHOOD_CONST, n_neighbors=None,   # Specify one of them. If `n_neighbors`
                 # is specified, `neighborhood_constant` will be ignored.
                 shared_nearest_neighbors=False,
                 metric=METRIC_DEF, metric_kwargs=None,        # distance metric and its parameter dict (if any)
                 approx_nearest_neighbors=True,
                 n_jobs=1,
                 reg_eps=0.001,
                 seed_rng=SEED_DEFAULT):
        """
        :param dim_projection: Dimension of data in the projected feature space. If set to 'auto', a suitable reduced
                               dimension will be chosen by estimating the intrinsic dimension of the data. If an
                               integer value is specified, it should be in the range `[1, dim - 1]`, where `dim`
                               is the observed dimension of the data.
        :param orthogonal: Set to True to select the OLPP method. It was shown to have better performance than LPP
                           in [3].
        :param pca_cutoff: float value in (0, 1] specifying the proportion of cumulative data variance to preserve
                           in the projected dimension-reduced data. PCA is applied as a first-level dimension
                           reduction to handle potential data matrix singularity also. Set `pca_cutoff = 1.0` in
                           order to handle only the data matrix singularity.
        :param neighborhood_constant: float value in (0, 1), that specifies the number of nearest neighbors as a
                                      function of the number of samples (data size). If `N` is the number of samples,
                                      then the number of neighbors is set to `N^neighborhood_constant`. It is
                                      recommended to set this value in the range 0.4 to 0.5.
        :param n_neighbors: None or int value specifying the number of nearest neighbors. If this value is specified,
                            the `neighborhood_constant` is ignored. It is sufficient to specify either
                            `neighborhood_constant` or `n_neighbors`.
        :param shared_nearest_neighbors: Set to True in order to use the shared nearest neighbor (SNN) distance to
                                         find the K nearest neighbors. This is a secondary distance metric that is
                                         found to be better suited to high dimensional data.
        :param metric: string or a callable that specifies the distance metric to be used for the SNN similarity
                       calculation.
        :param metric_kwargs: optional keyword arguments required by the distance metric specified in the form of a
                              dictionary.
        :param approx_nearest_neighbors: Set to True in order to use an approximate nearest neighbor algorithm to
                                         find the nearest neighbors. This is recommended when the number of points is
                                         large and/or when the dimension of the data is high.
        :param n_jobs: Number of parallel jobs or processes. Set to -1 to use all the available cpu cores.
        :param reg_eps: small float value that multiplies the trace to regularize the Gram matrix, if it is
                        close to singular.
        :param seed_rng: int value specifying the seed for the random number generator.
        """
        self.dim_projection = dim_projection
        self.orthogonal = orthogonal
        self.pca_cutoff = pca_cutoff
        self.neighborhood_constant = neighborhood_constant
        self.n_neighbors = n_neighbors
        self.shared_nearest_neighbors = shared_nearest_neighbors
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.approx_nearest_neighbors = approx_nearest_neighbors
        self.n_jobs = get_num_jobs(n_jobs)
        self.reg_eps = reg_eps
        self.seed_rng = seed_rng

        self.mean_data = None
        self.index_knn = None
        self.iterated_laplacian_matrix = None
        self.transform_pca = None
        self.transform_npp = None
        self.transform_comb = None

    def fit(self, data):
        """
        Find the optimal projection matrix for the given data points.

        :param data: data matrix of shape `(N, d)` where `N` is the number of samples and `d` is the number of
                     dimensions.
        :return:
        """
        N, d = data.shape
        if self.n_neighbors is None:
            # Set number of nearest neighbors based on the data size and the neighborhood constant
            self.n_neighbors = int(np.ceil(N ** self.neighborhood_constant))

        logger.info("Applying PCA as first-level dimension reduction step")
        data, self.mean_data, self.transform_pca = pca_wrapper(data, cutoff=self.pca_cutoff,
                                                               seed_rng=self.seed_rng)

        # If `self.neighbors > data.shape[1]` (number of neighbors larger than the data dimension), then the
        # Gram matrix that comes up while solving for the neighborhood weights becomes singular. To avoid this,
        # we can set `self.neighbors = data.shape[1]` or add a small nonzero value to the diagonal elements of the
        # Gram matrix
        d = data.shape[1]
        if self.n_neighbors >= d:
            k = max(d - 1, 1)
            logger.info("Reducing the number of neighbors from {:d} to {:d} to avoid singular Gram "
                        "matrix while solving for neighborhood weights.".format(self.n_neighbors, k))
            self.n_neighbors = k

        if self.dim_projection == 'auto':
            # Estimate the intrinsic dimension of the data and use that as the projected dimension
            id = estimate_intrinsic_dimension(data,
                                              method='two_nn',
                                              n_neighbors=self.n_neighbors,
                                              approx_nearest_neighbors=self.approx_nearest_neighbors,
                                              n_jobs=self.n_jobs,
                                              seed_rng=self.seed_rng)
            self.dim_projection = int(np.ceil(id))
            logger.info("Estimated intrinsic dimension of the (PCA-projected) data = {:.2f}.".format(id))

        if self.dim_projection >= data.shape[1]:
            self.dim_projection = data.shape[1]

        logger.info("Dimension of the projected subspace = {:d}".format(self.dim_projection))

        # Create a KNN index for all nearest neighbor tasks
        self.index_knn = KNNIndex(
            data, n_neighbors=self.n_neighbors,
            metric=self.metric, metric_kwargs=self.metric_kwargs,
            shared_nearest_neighbors=self.shared_nearest_neighbors,
            approx_nearest_neighbors=self.approx_nearest_neighbors,
            n_jobs=self.n_jobs,
            seed_rng=self.seed_rng
        )

        # Create the adjacency matrix `W` based on the optimal reconstruction weights of neighboring points
        # (as done in locally linear embedding).
        # Then calculate the iterated graph Laplacian matrix `M = (I - W)^T (I - W)`.
        self.create_iterated_laplacian(data)

        # Solve the generalized eigenvalue problem and take the eigenvectors corresponding to the smallest
        # eigenvalues as the columns of the projection matrix
        logger.info("Solving the generalized eigenvalue problem to find the optimal projection matrix.")
        data_trans = data.T
        # X^T M X
        lmat = sparse.csr_matrix.dot(data_trans, self.iterated_laplacian_matrix).dot(data)
        if self.orthogonal:
            # ONPP, the paper [2] recommends skipping the eigenvector corresponding to the smallest eigenvalue
            eig_values, eig_vectors = eigh(lmat, eigvals=(1, self.dim_projection))
        else:
            # Standard NPP or NPE
            # X^T X
            rmat = np.dot(data_trans, data)
            eig_values, eig_vectors = eigh(lmat, b=rmat, eigvals=(0, self.dim_projection - 1))

        # `eig_vectors` is a numpy array with each eigenvector along a column. The eigenvectors are ordered
        # according to increasing eigenvalues.
        # `eig_vectors` will have shape `(data.shape[1], self.dim_projection)`
        self.transform_npp = eig_vectors

        self.transform_comb = np.dot(self.transform_pca, self.transform_npp)

    def transform(self, data, dim=None):
        """
        Transform the given data by first subtracting the mean and then applying the linear projection.
        Optionally, you can specify the dimension of the transformed data using `dim`. This cannot be larger
        than `self.dim_projection`.

        :param data: numpy array of shape `(N, d)` with `N` samples in `d` dimensions.
        :param dim: If set to `None`, the dimension of the transformed data is `self.dim_projection`.
                    Else `dim` can be set to a value <= `self.dim_projection`. Doing this basically takes only
                    the `dim` top eigenvectors.

        :return:
            - data_trans: numpy array of shape `(N, dim)` with the transformed, dimension-reduced data.
        """
        if dim is None:
            data_trans = np.dot(data - self.mean_data, self.transform_comb)
        else:
            data_trans = np.dot(data - self.mean_data, self.transform_comb[:, 0:dim])

        return data_trans

    def fit_transform(self, data, dim=None):
        """
        Fit the model and transform the given data.

        :param data: numpy array of shape `(N, d)` with `N` samples in `d` dimensions.
        :param dim: same as the `transform` method.

        :return:
            - data_trans: numpy array of shape `(N, dim)` with the transformed, dimension-reduced data.
        """
        self.fit(data)
        return self.transform(data, dim=dim)

    def create_iterated_laplacian(self, data):
        """
        Calculate the optimal edge weights corresponding to the nearest neighbors of each point. This is exactly
        the same as the first step of the locally linear embedding (LLE) method.

        :param data: numpy array of shape `(N, d)` with `N` samples in `d` dimensions.
        :return: None
        """
        # Find the `self.n_neighbors` nearest neighbors of each point
        nn_indices, nn_distances = self.index_knn.query_self(k=self.n_neighbors)
        N, K = nn_indices.shape

        if self.n_jobs == 1:
            w = [helper_solve_lle(data, nn_indices, self.reg_eps, i) for i in range(N)]
        else:
            helper_partial = partial(helper_solve_lle, data, nn_indices, self.reg_eps)
            pool_obj = multiprocessing.Pool(processes=self.n_jobs)
            w = []
            _ = pool_obj.map_async(helper_partial, range(N), callback=w.extend)
            pool_obj.close()
            pool_obj.join()

        # Create a sparse matrix of size `(N, N)` for the adjacency matrix
        row_ind = np.array([[i] * (K + 1) for i in range(N)], dtype=np.int).ravel()
        col_ind = np.insert(nn_indices, 0, np.arange(N), axis=1).ravel()
        w = np.negative(w)
        vals = np.insert(w, 0, 1.0, axis=1).ravel()
        # Matrix `I - W`
        mat_tmp = sparse.csr_matrix((vals, (row_ind, col_ind)), shape=(N, N))

        # Matrix `M = (I - W)^T (I - W)`, also referred to as the iterated graph Laplacian
        self.iterated_laplacian_matrix = sparse.csr_matrix.dot(mat_tmp.transpose(), mat_tmp)


def solve_lle_weights(x, neighbors, reg_eps=0.001):
    """
    Solve for the optimal weights that reconstruct a point using its nearest neighbors. The weights are constrained
    to be non-negative and sum to 1. This is the first step of locally linear embedding (LLE).

    :param x: numpy array of shape `(dim, )`, where `dim` is the feature dimension.
    :param neighbors: numpy array of shape `(k, dim)`, where `k` is the number of neighbors.
    :param reg_eps: value close to 0 that is used to regularize any singular matrix.
    :return: numpy array with the optimal weights of shape `(k, )`.
    """
    # Gram matrix of the neighborhood of the point
    Z = x - neighbors
    G = np.dot(Z, Z.T)
    k = G.shape[0]

    eps = sys.float_info.epsilon
    # Trace of G is the sum of squared distance from `x` to its neighbors
    G_diag = G.diagonal()
    tr_G = np.sum(G_diag)
    max_diag_G = np.max(G_diag)
    if tr_G < eps or max_diag_G < eps:
        # Equal weight to all the neighbors
        return (1. / k) * np.ones(k)

    # Normalize the diagonal values to the range [0, 1]. Then look for really small values
    mask = G_diag < (eps * max_diag_G)
    if np.any(mask):
        # One or more of the neighbors are very close to the point `x`. In this case, we assign equal weight to
        # such overlapping neighbors
        w = np.zeros(k)
        w[mask] = 1.
        return w / np.sum(w)

    # Check for large value of condition number of `G`
    if np.linalg.cond(G) > 1e6:
        # Add a small perturbation to the diagonal of `G` to ensure that it is not singular
        np.fill_diagonal(G, G_diag + reg_eps * tr_G)

    # Solve for the optimal weights, ensure they are non-negative, and normalize them to sum to 1
    w = solve(G, np.ones(k), assume_a='pos')
    w = np.clip(w, 0., None)
    return w / np.sum(w)


def helper_solve_lle(data, nn_indices, reg_eps, n):
    return solve_lle_weights(data[n, :], data[nn_indices[n, :], :], reg_eps=reg_eps)


def helper_reconstruction_error(data, data_knn, nn_indices, reg_eps, n):
    # Euclidean norm of the error in the LLE reconstruction
    x = data[n, :]
    # nearest neighbors of `x`
    y = data_knn[nn_indices[n, :], :]
    w = solve_lle_weights(x, y, reg_eps=reg_eps)
    e = x - np.dot(w, y)

    return np.sum(e ** 2) ** 0.5


def transform_data_from_model(data, model_dict):
    """
    Given the mean vector and the projection matrix, transform (project) the input data.

    :param data: numpy array of shape `(N, D)`, where `N` is the number of samples and `D` the original dimension.
    :param model_dict: dict with the transform parameters.

    :return: transformed data as a numpy array of shape `(N, d)`, where `d` is the reduced dimension.
    """
    if model_dict is None:
        return data
    else:
        return np.dot(data - model_dict['mean_data'], model_dict['transform'])


def load_dimension_reduction_models(model_file):
    """
    :param model_file: model file name.

    :return: list of model dicts, one per DNN layer. Each model dict is of the form:
        'method': <name of the method used>
        'mean_data': <1D numpy array with the mean of the data features>
        'transform': <2D numpy array with the transformation matrix>
    """
    with open(model_file, 'rb') as fp:
        models = pickle.load(fp)

    return models


def wrapper_data_projection(data, method,
                            data_test=None,
                            dim_proj=3,
                            metric=METRIC_DEF, metric_kwargs=None,
                            snn=False,
                            ann=True,
                            pca_cutoff=1.0,
                            n_jobs=1,
                            seed_rng=SEED_DEFAULT,
                            model_filename=None):
    """
    Wrapper function to apply different dimension reduction methods. The reduced dimension can be set either to a
    single value or a list (or iterable) of values via the argument `dim_proj`. If the reduced dimension is to be
    varied over a range of values, it is more efficient to find the optimal transformation corresponding to the
    largest of the (reduced dimension) values once, and then project (transform) the data by taking the
    required number of columns of the transformation matrix.

    :param data: numpy array of shape `(N, d)` with the training data. `N` and `d` are the number of samples
                 and features respectively.
    :param method: one of the methods `['LPP', 'OLPP', 'NPP', 'ONPP', 'PCA']`.
    :param data_test: None or a numpy array of test data similar to the input `data`.
    :param dim_proj: int or an iterable of int values. If `int`, this is taken as the dimension of the projected
                     data. If an iterable of int values is specifed, the data is projected into a reduced dimension
                     space corresponding to each value. The returned values will be different in this case.
    :param metric: distance metric which can be a predefined string or a callable.
    :param metric_kwargs: None or a dict of keyword arguments required by the metric callable.
    :param snn: set to True to use shared nearest neighbors.
    :param ann: set to True to use approximate nearest neighbors.
    :param pca_cutoff: variance cutoff value in (0, 1]. This value is used to select the number of components
                       only if `n_comp` is not specified.
    :param n_jobs: number of cpu cores to use for parallel processing.
    :param seed_rng: seed for the random number generator.
    :param model_filename: (str) filename to save the model for reuse. The file is saved in pickle format. By default,
                           this is set to None and the model is not saved.

    :return:
        - model_dict: dict with the model transform parameters.
        - data_proj: dimension reduced version of `data`. If `dim_proj` is an integer value, this will be a numpy
                     array of shape `(N, dim_proj)`. If `dim_proj` is an iterable of integer values, then this will
                     be a list of numpy arrays, where each numpy array is the transformed data corresponding to a
                     single value in `dim_proj`.
        - data_proj_test: Returned only if `data_test is not None`.
                          This will be a dimension reduced version of `data_test`. Same format as `data_proj`.
    """
    # Choices are {'simple', 'SNN', 'heat_kernel'}
    edge_weights = 'heat_kernel'

    # Neighborhood size is chosen as a function of the number of data points
    neighborhood_constant = NEIGHBORHOOD_CONST

    if hasattr(dim_proj, '__iter__'):
        dim_proj_max = max(dim_proj)
        rtype = 'list'
    else:
        dim_proj_max = dim_proj
        rtype = 'arr'

    rtest = (data_test is not None)
    data_proj = None
    data_proj_test = None
    model_dict = None
    if method == 'LPP' or method == 'OLPP':
        model = LocalityPreservingProjection(
            dim_projection=dim_proj_max,
            orthogonal=(method == 'OLPP'),
            pca_cutoff=pca_cutoff,
            neighborhood_constant=neighborhood_constant,
            shared_nearest_neighbors=snn,
            edge_weights=edge_weights,
            metric=metric,
            metric_kwargs=metric_kwargs,
            approx_nearest_neighbors=ann,
            n_jobs=n_jobs,
            seed_rng=seed_rng
        )
        model.fit(data)
        data_proj = model.transform(data)
        if rtest:
            data_proj_test = model.transform(data_test)

        d_max = data_proj.shape[1]
        if rtype == 'list':
            data_proj = [data_proj[:, :d] for d in dim_proj if d <= d_max]
            if rtest:
                data_proj_test = [data_proj_test[:, :d] for d in dim_proj if d <= d_max]

        model_dict = {'method': method, 'mean_data': model.mean_data, 'transform': model.transform_comb}
    elif method == 'NPP' or method == 'ONPP':
        model = NeighborhoodPreservingProjection(
            dim_projection=dim_proj_max,
            orthogonal=(method == 'ONPP'),
            pca_cutoff=pca_cutoff,
            neighborhood_constant=neighborhood_constant,
            shared_nearest_neighbors=snn,
            metric=metric,
            metric_kwargs=metric_kwargs,
            approx_nearest_neighbors=ann,
            n_jobs=n_jobs,
            seed_rng=seed_rng
        )
        model.fit(data)
        data_proj = model.transform(data)
        if rtest:
            data_proj_test = model.transform(data_test)

        d_max = data_proj.shape[1]
        if rtype == 'list':
            data_proj = [data_proj[:, :d] for d in dim_proj if d <= d_max]
            if rtest:
                data_proj_test = [data_proj_test[:, :d] for d in dim_proj if d <= d_max]

        model_dict = {'method': method, 'mean_data': model.mean_data, 'transform': model.transform_comb}
    elif method == 'PCA':
        data_proj, mean_data, transform_pca = pca_wrapper(data, n_comp=dim_proj_max, cutoff=pca_cutoff,
                                                          seed_rng=seed_rng)
        if rtest:
            data_proj_test = np.dot(data_test - mean_data, transform_pca)

        d_max = data_proj.shape[1]
        if rtype == 'list':
            data_proj = [data_proj[:, :d] for d in dim_proj if d <= d_max]
            if rtest:
                data_proj_test = [data_proj_test[:, :d] for d in dim_proj if d <= d_max]

        model_dict = {'method': method, 'mean_data': mean_data, 'transform': transform_pca}
    else:
        raise ValueError("Invalid value '{}' received by parameter 'method'".format(method))

    if model_filename:
        with open(model_filename, 'wb') as fp:
            pickle.dump(model_dict, fp)

    if rtest:
        return model_dict, data_proj, data_proj_test
    else:
        return model_dict, data_proj
