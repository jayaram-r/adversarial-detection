"""
Implementation of the deep k-nearest neighbors paper:

Papernot, Nicolas, and Patrick McDaniel. "Deep k-nearest neighbors: Towards confident, interpretable and robust
deep learning." arXiv preprint arXiv:1803.04765 (2018).

The credibility score - maximum p-value across classes - is used for the detection of out-of-distribution and
adversarial samples. Samples with low credibility are likely to be out-of-distribution or adversarial.

While their paper used the locality sensitive hashing method for approximate nearest neighbors (ANN), we use the
NN-descent method with the cosine distance. Also, we allow the option of performing dimensionality reduction on
the DNN layer representations using the neighborhood preserving projection (NPP) method.

"""
import numpy as np
import sys
import logging
from helpers.constants import (
    SEED_DEFAULT,
    NEIGHBORHOOD_CONST,
    METRIC_DEF
)
from helpers.dimension_reduction_methods import (
    transform_data_from_model,
    load_dimension_reduction_models
)
from helpers.knn_index import KNNIndex
from helpers.knn_classifier import neighbors_label_counts
from helpers.utils import get_num_jobs

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


class DeepKNN:
    """
    Implementation of the deep k-nearest neighbors paper:

    Papernot, Nicolas, and Patrick McDaniel. "Deep k-nearest neighbors: Towards confident, interpretable and robust
    deep learning." arXiv preprint arXiv:1803.04765 (2018).
    """
    _name = 'dknn'
    def __init__(self,
                 neighborhood_constant=NEIGHBORHOOD_CONST, n_neighbors=None,
                 metric=METRIC_DEF, metric_kwargs=None,
                 approx_nearest_neighbors=True,
                 skip_dim_reduction=True,
                 model_dim_reduction=None,
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
        :param metric: string or a callable that specifies the distance metric to use.
        :param metric_kwargs: optional keyword arguments required by the distance metric specified in the form of a
                              dictionary.
        :param approx_nearest_neighbors: Set to True in order to use an approximate nearest neighbor algorithm to
                                         find the nearest neighbors. The NN-descent method is used for approximate
                                         nearest neighbor searches.
        :param skip_dim_reduction: Set to True in order to skip dimension reduction of the layer embeddings.
        :param model_dim_reduction: 1. None if dimension reduction is not required; (OR)
                                    2. Path to a file containing the saved dimension reduction model. This will be
                                       a pickle file that loads into a list of model dictionaries; (OR)
                                    3. The dimension reduction model loaded into memory from the pickle file.
        :param n_jobs: Number of parallel jobs or processes. Set to -1 to use all the available cpu cores.
        :param low_memory: Set to True to enable the low memory option of the `NN-descent` method. Note that this
                           is likely to increase the running time.
        :param seed_rng: int value specifying the seed for the random number generator. This is passed around to
                         all the classes/functions that require random number generation. Set this to a fixed value
                         for reproducible results.
        """
        self.neighborhood_constant = neighborhood_constant
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.approx_nearest_neighbors = approx_nearest_neighbors
        self.skip_dim_reduction = skip_dim_reduction
        self.n_jobs = get_num_jobs(n_jobs)
        self.low_memory = low_memory
        self.seed_rng = seed_rng

        np.random.seed(self.seed_rng)
        # Load the dimension reduction models per-layer if required
        self.transform_models = None
        if not self.skip_dim_reduction:
            if model_dim_reduction is None:
                raise ValueError("Model file for dimension reduction is required but not specified as input.")
            elif isinstance(model_dim_reduction, str):
                # Pickle file is specified
                self.transform_models = load_dimension_reduction_models(model_dim_reduction)
            elif isinstance(model_dim_reduction, list):
                # Model already loaded from pickle file
                self.transform_models = model_dim_reduction
            else:
                raise ValueError("Invalid format for the dimension reduction model input.")

        self.n_layers = None
        self.labels_unique = None
        self.n_classes = None
        self.n_samples = None
        self.label_encoder = None
        # Encoded labels of train data
        self.labels_train_enc = None
        # KNN index for data from each layer
        self.index_knn = None
        self.mask_exclude = None
        # Non-conformity values on the calibration data
        self.nonconformity_calib = None

    def fit(self, layer_embeddings, labels):
        """
        Estimate parameters of the detection method given natural (non-adversarial) input data. Note that this
        data should be different from that used to train the DNN classifier.
        NOTE: Inputs to this method can be obtained by calling the function `extract_layer_embeddings`.

        :param layer_embeddings: list of numpy arrays with the layer embedding data. Length of the list is equal to
                                 the number of layers. The numpy array at index `i` has shape `(n, d_i)`, where `n`
                                 is the number of samples and `d_i` is the dimension of the embeddings at layer `i`.
        :param labels: numpy array of labels for the classification problem addressed by the DNN. Should have shape
                       `(n, )`, where `n` is the number of samples.
        :return: Instance of the class with all parameters fit to the data.
        """
        self.n_layers = len(layer_embeddings)
        self.labels_unique = np.unique(labels)
        self.n_classes = len(self.labels_unique)
        self.n_samples = labels.shape[0]
        # Mapping from the original labels to the set {0, 1, . . .,self.n_classes - 1}. This is needed by the label
        # count function
        d = dict(zip(self.labels_unique, np.arange(self.n_classes)))
        self.label_encoder = np.vectorize(d.__getitem__)

        # Number of nearest neighbors
        if self.n_neighbors is None:
            # Set number of nearest neighbors based on the data size and the neighborhood constant
            self.n_neighbors = int(np.ceil(self.n_samples ** self.neighborhood_constant))

        logger.info("Number of classes: {:d}.".format(self.n_classes))
        logger.info("Number of layer embeddings: {:d}.".format(self.n_layers))
        logger.info("Number of samples: {:d}.".format(self.n_samples))
        logger.info("Number of neighbors: {:d}.".format(self.n_neighbors))
        if not all([layer_embeddings[i].shape[0] == self.n_samples for i in range(self.n_layers)]):
            raise ValueError("Input 'layer_embeddings' does not have the expected format")

        self.labels_train_enc = self.label_encoder(labels)
        indices_true = dict()
        self.mask_exclude = np.ones((self.n_classes, self.n_classes), dtype=np.bool)
        for j, c in enumerate(self.labels_unique):
            # Index of labeled samples from class `c`
            indices_true[c] = np.where(labels == c)[0]
            self.mask_exclude[j, j] = False

        self.nonconformity_calib = np.zeros(self.n_samples)
        self.index_knn = [None for _ in range(self.n_layers)]
        for i in range(self.n_layers):
            logger.info("Processing layer {:d}:".format(i + 1))
            if self.transform_models:
                logger.info("Transforming the embeddings from layer {:d}.".format(i + 1))
                data_proj = transform_data_from_model(layer_embeddings[i], self.transform_models[i])
                logger.info("Input dimension = {:d}, projected dimension = {:d}".
                            format(layer_embeddings[i].shape[1], data_proj.shape[1]))
            else:
                data_proj = layer_embeddings[i]

            logger.info("Building a KNN index for nearest neighbor queries.")
            # Build a KNN index on the set of feature embeddings from normal samples from layer `i`
            self.index_knn[i] = KNNIndex(
                data_proj, n_neighbors=self.n_neighbors,
                metric=self.metric, metric_kwargs=self.metric_kwargs,
                approx_nearest_neighbors=self.approx_nearest_neighbors,
                n_jobs=self.n_jobs,
                low_memory=self.low_memory,
                seed_rng=self.seed_rng
            )
            # Indices of the nearest neighbors of each sample
            nn_indices, _ = self.index_knn[i].query_self(k=self.n_neighbors)
            logger.info("Calculating the class label counts and non-conformity scores in the neighborhood of "
                        "each sample.")
            _, nc_counts = neighbors_label_counts(nn_indices, self.labels_train_enc, self.n_classes)

            for j, c in enumerate(self.labels_unique):
                # Neighborhood counts of all classes except `c`
                nc_counts_slice = nc_counts[:, self.mask_exclude[j, :]]
                # Nonconformity from layer `i` for all labeled samples from class `c`
                self.nonconformity_calib[indices_true[c]] += np.sum(nc_counts_slice[indices_true[c], :], axis=1)

        return self

    def score(self, layer_embeddings, is_train=False):
        """
        :param layer_embeddings: list of numpy arrays with the layer embedding data. Length of the list is equal to
                                 the number of layers. The numpy array at index `i` has shape `(n, d_i)`, where `n`
                                 is the number of samples and `d_i` is the dimension of the embeddings at layer `i`.
        :param is_train: Set to True if the inputs are the same non-adversarial inputs used with the `fit` method.

        :return: (scores, predictions)
            - scores: numpy array of scores corresponding to OOD or adversarial detection. It is the negative log
                      of the credibility scores. So high values of this score correspond to low credibility (i.e.
                      high probability of an outlier).
            - predictions: numpy array of the corrected deep kNN class predictions. Has the same shape as `scores`.
        """
        n_test = layer_embeddings[0].shape[0]
        l = len(layer_embeddings)
        if l != self.n_layers:
            raise ValueError("Expecting {:d} layers in the input data, but received {:d}".format(self.n_layers, l))

        nonconformity_per_class = np.zeros((n_test, self.n_classes))
        for i in range(self.n_layers):
            if self.transform_models:
                # Dimension reduction
                data_proj = transform_data_from_model(layer_embeddings[i], self.transform_models[i])
            else:
                data_proj = layer_embeddings[i]

            # Indices of the nearest neighbors of each test sample
            if is_train:
                nn_indices, _ = self.index_knn[i].query_self(k=self.n_neighbors)
            else:
                nn_indices, _ = self.index_knn[i].query(data_proj, k=self.n_neighbors)

            # Class label counts among the nearest neighbors
            _, nc_counts = neighbors_label_counts(nn_indices, self.labels_train_enc, self.n_classes)

            for j, c in enumerate(self.labels_unique):
                # Neighborhood counts of all classes except `c`
                nc_counts_slice = nc_counts[:, self.mask_exclude[j, :]]
                # Nonconformity w.r.t class `c` from layer `i`
                nonconformity_per_class[:, j] += np.sum(nc_counts_slice, axis=1)

        # Calculate the p-values per-class with respect to the non-conformity scores of the calibration set
        mask = self.nonconformity_calib[:, np.newaxis] >= nonconformity_per_class.ravel()[np.newaxis, :]
        v = np.sum(mask, axis=0) / float(mask.shape[0])
        p_values = v.reshape((n_test, self.n_classes))
        # Credibility is the maximum p-value over all classes
        credibility = np.max(p_values, axis=1)
        # Anomaly score
        scores = -np.log(np.clip(credibility, sys.float_info.min, None))
        # Deep k-NN prediction is the class corresponding to the largest p-value
        predictions = np.array([self.labels_unique[j] for j in np.argmax(p_values, axis=1)],
                               dtype=self.labels_unique.dtype)

        return scores, predictions
