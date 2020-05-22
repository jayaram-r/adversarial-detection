"""
Implementation of the adversarial attack detection method from the paper:

Ma, Xingjun, et al. "Characterizing adversarial subspaces using local intrinsic dimensionality.",
 International conference on learning representations, 2018.
https://arxiv.org/pdf/1801.02613.pdf

Some of this code is borrowed from: https://github.com/xingjunm/lid_adversarial_subspace_detection

Note that this implementation does not use the mini-batching method to estimate LID as done in the paper.
Since the main utility of mini-batching was for computational efficiency, we instead use the approximate nearest
neighbors method for fast querying of neighbors from the full set of normal data.

We also implement an extension of their method (see class `DetectorLIDClassCond`) that estimates LID values (at each
layer) specific to each class manifold. For a test sample, the LID estimate is based on the non-adversarial samples
from its predicted class. This method improves the performance of the detector significantly.

Other notes on the implementation:
- Min-max scaling of LID features used as done in their code repo
- 5-fold stratified cross-validation used to search for the best weight regularization coefficient.
- Area under the ROC curve is used as the metric for cross-validation.
- To handle class imbalance (adversarial vs. non-adversarial), the implmentation allows the classes to be balanced
by assigning sample weights.

"""
import numpy as np
import sys
import logging
import os
import tempfile
import subprocess
from helpers.constants import (
    SEED_DEFAULT,
    NEIGHBORHOOD_CONST,
    CROSS_VAL_SIZE,
    METRIC_DEF
)
from helpers.dimension_reduction_methods import (
    transform_data_from_model,
    load_dimension_reduction_models
)
from helpers.knn_index import KNNIndex, helper_knn_distance
from helpers.lid_estimators import lid_mle_amsaleg
from helpers.utils import get_num_jobs
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import MinMaxScaler
try:
    import cPickle as pickle
except:
    import pickle

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


class DetectorLID:
    _name = 'lid'
    def __init__(self,
                 neighborhood_constant=NEIGHBORHOOD_CONST, n_neighbors=None,
                 metric='euclidean', metric_kwargs=None,
                 n_cv_folds=CROSS_VAL_SIZE,
                 c_search_values=None,
                 approx_nearest_neighbors=True,
                 skip_dim_reduction=True,
                 model_dim_reduction=None,
                 n_jobs=1,
                 max_iter=200,
                 balanced_classification=True,
                 low_memory=False,
                 save_knn_indices_to_file=True,
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
        :param n_cv_folds: number of cross-validation folds.
        :param c_search_values: list or array of search values for the logistic regression hyper-parameter `C`. The
                                default value is `None`.
        :param approx_nearest_neighbors: Set to True in order to use an approximate nearest neighbor algorithm to
                                         find the nearest neighbors. The NN-descent method is used for approximate
                                         nearest neighbor searches.
        :param skip_dim_reduction: Set to True in order to skip dimension reduction of the layer embeddings.
        :param model_dim_reduction: 1. None if dimension reduction is not required; (OR)
                                    2. Path to a file containing the saved dimension reduction model. This will be
                                       a pickle file that loads into a list of model dictionaries; (OR)
                                    3. The dimension reduction model loaded into memory from the pickle file.
        :param n_jobs: Number of parallel jobs or processes. Set to -1 to use all the available cpu cores.
        :param max_iter: Maximum number of iterations for the optimization of the logistic classifier. The default
                         value set by the scikit-learn library is 100, but sometimes this does not allow for
                         convergence. Hence, increasing it to 200 here.
        :param balanced_classification: Set to True to assign sample weights to balance the binary classification
                                        problem separating adversarial from non-adversarial samples.
        :param low_memory: Set to True to enable the low memory option of the `NN-descent` method. Note that this
                           is likely to increase the running time.
        :param save_knn_indices_to_file: Set to True in order to save the KNN indices from each layer to a pickle
                                         file to reduce memory usage. This may not be needed when the data size
                                         and/or the number of layers is small. It avoids potential out-of-memory
                                         errors at the expense of time taken to write and read the files.
        :param seed_rng: int value specifying the seed for the random number generator. This is passed around to
                         all the classes/functions that require random number generation. Set this to a fixed value
                         for reproducible results.
        """
        self.neighborhood_constant = neighborhood_constant
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.n_cv_folds = n_cv_folds
        self.c_search_values = c_search_values
        self.approx_nearest_neighbors = approx_nearest_neighbors
        self.skip_dim_reduction = skip_dim_reduction
        self.n_jobs = get_num_jobs(n_jobs)
        self.max_iter = max_iter
        self.balanced_classification = balanced_classification
        self.low_memory = low_memory
        self.save_knn_indices_to_file = save_knn_indices_to_file
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

        if self.c_search_values is None:
            # Default search values for the `C` parameter of logistic regression
            self.c_search_values = np.logspace(-4, 4, num=10)

        self.n_layers = None
        self.n_samples = []
        self.index_knn = None
        self.model_logistic = None
        self.scaler = None
        # Temporary directory to save the KNN index files
        self.temp_direc = None
        self.temp_knn_files = None

    def fit(self, layer_embeddings_normal, layer_embeddings_adversarial, layer_embeddings_noisy=None):
        """
        Extract the LID feature vector for normal, noisy, and adversarial samples and train a logistic classifier
        to separate adversarial samples from (normal + noisy). Cross-validation is used to select the hyper-parameter
        `C` using area under the ROC curve as the validation metric.

        :param layer_embeddings_normal: list of numpy arrays with the layer embeddings for normal samples.
                                        Length of the list is equal to the number of layers. The numpy array at
                                        index `i` has shape `(n, d_i)`, where `n` is the number of samples and `d_i`
                                        is the dimension of the embeddings at layer `i`.
        :param layer_embeddings_adversarial: Same format as `layer_embeddings_normal`, but corresponding to
                                             adversarial data.
        :param layer_embeddings_noisy: Same format as `layer_embeddings_normal`, but corresponding to noisy data.
                                       Can be set to `None` to exclude noisy data from training.
        :return:
            (self, scores_normal, scores_adversarial) if layer_embeddings_noise is None
            (self, scores_normal, scores_adversarial, scores_noisy) otherwise.
            -------------------------------------------------------
            - self: trained instance of the class.
            - scores_normal: numpy array with the scores (decision function of the logistic classifier) for normal
                             samples. 1d array with the same number of samples as `layer_embeddings_normal`.
            - scores_noisy: scores corresponding to `layer_embeddings_noisy` if noisy training data is provided.
            - scores_adversarial: scores corresponding to `layer_embeddings_adversarial`.
        """
        self.n_layers = len(layer_embeddings_normal)
        logger.info("Number of layer embeddings: {:d}.".format(self.n_layers))
        if layer_embeddings_noisy is None:
            logger.info("Noisy training data not provided.")
            cond1 = False
            noisy_data = False
        else:
            cond1 = (len(layer_embeddings_noisy) != self.n_layers)
            noisy_data = True

        if cond1 or (len(layer_embeddings_adversarial) != self.n_layers):
            raise ValueError("The layer embeddings for noisy and attack samples must have the same length as that "
                             "of normal samples")

        # Number of samples in each of the categories
        self.n_samples = [
            layer_embeddings_normal[0].shape[0],
            layer_embeddings_noisy[0].shape[0] if noisy_data else 0,
            layer_embeddings_adversarial[0].shape[0]
        ]
        # Number of nearest neighbors
        if self.n_neighbors is None:
            # Set number of nearest neighbors based on the data size (of normal samples) and the
            # neighborhood constant
            self.n_neighbors = int(np.ceil(self.n_samples[0] ** self.neighborhood_constant))

        # The data arrays at all layers should have the same number of samples
        if not all([layer_embeddings_normal[i].shape[0] == self.n_samples[0] for i in range(self.n_layers)]):
            raise ValueError("Input 'layer_embeddings_normal' does not have the expected format")

        if noisy_data:
            if not all([layer_embeddings_noisy[i].shape[0] == self.n_samples[1] for i in range(self.n_layers)]):
                raise ValueError("Input 'layer_embeddings_noisy' does not have the expected format")

        if not all([layer_embeddings_adversarial[i].shape[0] == self.n_samples[2] for i in range(self.n_layers)]):
            raise ValueError("Input 'layer_embeddings_adversarial' does not have the expected format")

        if self.save_knn_indices_to_file:
            # Create a temporary directory for saving the KNN indices
            self.temp_direc = tempfile.mkdtemp(dir=os.getcwd())
            self.temp_knn_files = [''] * self.n_layers

        self.index_knn = [None for _ in range(self.n_layers)]
        features_lid_normal = np.zeros((self.n_samples[0], self.n_layers))
        features_lid_noisy = np.zeros((self.n_samples[1], self.n_layers))
        features_lid_adversarial = np.zeros((self.n_samples[2], self.n_layers))
        for i in range(self.n_layers):
            logger.info("Processing layer {:d}:".format(i + 1))
            if self.transform_models:
                data_normal = transform_data_from_model(layer_embeddings_normal[i], self.transform_models[i])
                data_adver = transform_data_from_model(layer_embeddings_adversarial[i], self.transform_models[i])
                if noisy_data:
                    data_noisy = transform_data_from_model(layer_embeddings_noisy[i], self.transform_models[i])
                else:
                    data_noisy = None

                d1 = layer_embeddings_normal[i].shape[1]
                d2 = data_normal.shape[1]
                if d2 < d1:
                    logger.info("Input dimension = {:d}, projected dimension = {:d}".format(d1, d2))
            else:
                data_normal = layer_embeddings_normal[i]
                data_adver = layer_embeddings_adversarial[i]
                if noisy_data:
                    data_noisy = layer_embeddings_noisy[i]
                else:
                    data_noisy = None

            logger.info("Building a KNN index on the feature embeddings of normal samples.")
            # Build a KNN index on the set of feature embeddings from normal samples from layer `i`
            self.index_knn[i] = KNNIndex(
                data_normal, n_neighbors=self.n_neighbors,
                metric=self.metric, metric_kwargs=self.metric_kwargs,
                approx_nearest_neighbors=self.approx_nearest_neighbors,
                n_jobs=self.n_jobs,
                low_memory=self.low_memory,
                seed_rng=self.seed_rng
            )
            logger.info("Calculating LID estimates for the feature embeddings of normal samples.")
            # Nearest neighbors of the normal feature embeddings from this layer
            nn_indices, nn_distances = self.index_knn[i].query_self(k=self.n_neighbors)
            # LID estimates of the normal feature embeddings from this layer
            features_lid_normal[:, i] = lid_mle_amsaleg(nn_distances)

            if noisy_data:
                logger.info("Calculating LID estimates for the feature embeddings of noisy samples.")
                # Nearest neighbors of the noisy feature embeddings from this layer
                nn_indices, nn_distances = self.index_knn[i].query(data_noisy, k=self.n_neighbors)
                # LID estimates of the noisy feature embeddings from this layer
                features_lid_noisy[:, i] = lid_mle_amsaleg(nn_distances)

            logger.info("Calculating LID estimates for the feature embeddings of adversarial samples.")
            # Nearest neighbors of the adversarial feature embeddings from this layer
            nn_indices, nn_distances = self.index_knn[i].query(data_adver, k=self.n_neighbors)
            # LID estimates of the adversarial feature embeddings from this layer
            features_lid_adversarial[:, i] = lid_mle_amsaleg(nn_distances)

            if self.save_knn_indices_to_file:
                logger.info("Saving the KNN index from layer {:d} to a pickle file".format(i + 1))
                self.temp_knn_files[i] = os.path.join(self.temp_direc, 'knn_index_layer_{:d}.pkl'.format(i + 1))
                with open(self.temp_knn_files[i], 'wb') as fp:
                    pickle.dump(self.index_knn[i], fp)

                # Free up the allocated memory
                self.index_knn[i] = None

        # Feature vector and labels for the binary logistic classifier.
        # Normal and noisy samples are given labels 0 and adversarial samples are given label 1
        n_pos = features_lid_adversarial.shape[0]
        if noisy_data:
            features_lid = np.concatenate([features_lid_normal, features_lid_noisy, features_lid_adversarial],
                                          axis=0)
            labels = np.concatenate([np.zeros(features_lid_normal.shape[0], dtype=np.int),
                                     np.zeros(features_lid_noisy.shape[0], dtype=np.int),
                                     np.ones(n_pos, dtype=np.int)])
        else:
            features_lid = np.concatenate([features_lid_normal, features_lid_adversarial], axis=0)
            labels = np.concatenate([np.zeros(features_lid_normal.shape[0], dtype=np.int),
                                     np.ones(n_pos, dtype=np.int)])

        pos_prop = n_pos / float(labels.shape[0])
        # Randomly shuffle the samples to avoid determinism
        ind_perm = np.random.permutation(labels.shape[0])
        features_lid = features_lid[ind_perm, :]
        labels = labels[ind_perm]
        # Min-max scaling for the LID features
        self.scaler = MinMaxScaler().fit(features_lid)
        features_lid = self.scaler.transform(features_lid)
        logger.info("Training a binary logistic classifier with {:d} samples and {:d} LID features.".
                    format(*features_lid.shape))
        logger.info("Using {:d}-fold cross-validation with area under ROC curve as the metric to select "
                    "the best regularization hyperparameter.".format(self.n_cv_folds))
        logger.info("Proportion of positive (adversarial or OOD) samples in the training data: {:.4f}".
                    format(pos_prop))
        class_weight = None
        if self.balanced_classification:
            if (pos_prop < 0.45) or (pos_prop > 0.55):
                class_weight = {0: 1.0 / (1 - pos_prop),
                                1: 1.0 / pos_prop}
                logger.info("Balancing the classes by assigning sample weight {:.4f} to class 0 and sample weight "
                            "{:.4f} to class 1".format(class_weight[0], class_weight[1]))

        self.model_logistic = LogisticRegressionCV(
            Cs=self.c_search_values,
            cv=self.n_cv_folds,
            penalty='l2',
            scoring='roc_auc',
            multi_class='auto',
            class_weight=class_weight,
            max_iter=self.max_iter,
            refit=True,
            n_jobs=self.n_jobs,
            random_state=self.seed_rng
        ).fit(features_lid, labels)

        # Larger values of this score correspond to a higher probability of predicting class 1 (adversarial)
        scores_normal = self.model_logistic.decision_function(self.scaler.transform(features_lid_normal))
        scores_adversarial = self.model_logistic.decision_function(self.scaler.transform(features_lid_adversarial))
        if noisy_data:
            scores_noisy = self.model_logistic.decision_function(self.scaler.transform(features_lid_noisy))
            return self, scores_normal, scores_adversarial, scores_noisy
        else:
            return self, scores_normal, scores_adversarial

    def score(self, layer_embeddings, cleanup=True):
        """
        Given a list of layer embeddings for test samples, extract the layer-wise LID feature vector and return the
        decision function of the logistic classifier.

        :param layer_embeddings: list of numpy arrays with the layer embeddings for normal samples. Length of the
                                 list is equal to the number of layers. The numpy array at index `i` has shape
                                 `(n, d_i)`, where `n` is the number of samples and `d_i` is the dimension of the
                                 embeddings at layer `i`.
        :param cleanup: If set to True, the temporary directory where the KNN index files are saved will be deleted
                        after scoring. If this method is to be called multiple times, set `cleanup = False` for all
                        calls except the last one.
        :return:
            - numpy array of detection scores for the test samples. Has shape `(n, )` where `n` is the number of
              samples. Larger values correspond to a higher confidence that the sample is adversarial.
        """
        n_test = layer_embeddings[0].shape[0]
        l = len(layer_embeddings)
        if l != self.n_layers:
            raise ValueError("Expecting {:d} layers in the input 'layer_embeddings', but received {:d} layers.".
                             format(self.n_layers, l))

        features_lid = np.zeros((n_test, self.n_layers))
        for i in range(self.n_layers):
            logger.info("Calculating LID features for layer {:d}".format(i + 1))
            if self.transform_models:
                # Dimension reduction
                data_proj = transform_data_from_model(layer_embeddings[i], self.transform_models[i])
            else:
                data_proj = layer_embeddings[i]

            if self.save_knn_indices_to_file:
                with open(self.temp_knn_files[i], 'rb') as fp:
                    self.index_knn[i] = pickle.load(fp)

            _, nn_distances = self.index_knn[i].query(data_proj, k=self.n_neighbors)
            features_lid[:, i] = lid_mle_amsaleg(nn_distances)

            if self.save_knn_indices_to_file:
                self.index_knn[i] = None

        if cleanup and self.save_knn_indices_to_file:
            _ = subprocess.check_call(['rm', '-rf', self.temp_direc])

        features_lid = self.scaler.transform(features_lid)
        return self.model_logistic.decision_function(features_lid)


class DetectorLIDBatch:
    """
    A faster version of `DetectorLID` that randomly splits up the non-adversarial data into a fixed number of batches.
    It leverages the implementation of `DetectorLIDClassCond` by passing it randomly assigned labels and predicted
    labels.
    """
    _name = 'lid'
    def __init__(self,
                 n_batches=10,
                 neighborhood_constant=NEIGHBORHOOD_CONST, n_neighbors=None,
                 metric='euclidean', metric_kwargs=None,
                 n_cv_folds=CROSS_VAL_SIZE,
                 c_search_values=None,
                 approx_nearest_neighbors=True,
                 skip_dim_reduction=True,
                 model_dim_reduction=None,
                 n_jobs=1,
                 max_iter=200,
                 balanced_classification=True,
                 low_memory=False,
                 save_knn_indices_to_file=True,
                 seed_rng=SEED_DEFAULT):
        """

        :param n_batches: (int) number of batches to create out of the non-adversarial data.
        Rest of the parameters are the same as the class `DetectorLID`.
        """
        self.n_batches = n_batches
        np.random.seed(seed_rng)

        self.det_class_cond = DetectorLIDClassCond(
            neighborhood_constant=neighborhood_constant, n_neighbors=n_neighbors,
            metric=metric, metric_kwargs=metric_kwargs,
            n_cv_folds=n_cv_folds,
            c_search_values=c_search_values,
            approx_nearest_neighbors=approx_nearest_neighbors,
            skip_dim_reduction=skip_dim_reduction,
            model_dim_reduction=model_dim_reduction,
            n_jobs=n_jobs,
            max_iter=max_iter,
            balanced_classification=balanced_classification,
            low_memory=low_memory,
            save_knn_indices_to_file=save_knn_indices_to_file,
            seed_rng=seed_rng
        )

    def _gen_random_labels(self, n_samples):
        m = int(np.round(float(n_samples) / self.n_batches))
        arr = []
        for i in range(self.n_batches):
            arr.extend([i] * m)

        labels = np.array(arr[:n_samples], dtype=np.int)
        np.random.shuffle(labels)

        return labels

    def fit(self, layer_embeddings_normal, layer_embeddings_adversarial, layer_embeddings_noisy=None):
        """
        Same inputs and output as the `fit` method of the class `DetectorLID`.
        """
        n_normal = layer_embeddings_normal[0].shape[0]
        labels_normal = self._gen_random_labels(n_normal)
        labels_pred_normal = labels_normal

        labels_pred_noisy = None
        if layer_embeddings_noisy is not None:
            n_noisy = layer_embeddings_noisy[0].shape[0]
            if n_noisy == n_normal:
                labels_pred_noisy = labels_pred_normal
            else:
                labels_pred_noisy = self._gen_random_labels(n_noisy)

        n_adver = layer_embeddings_adversarial[0].shape[0]
        labels_pred_adver = self._gen_random_labels(n_adver)

        return self.det_class_cond.fit(layer_embeddings_normal, labels_normal, labels_pred_normal,
                                       layer_embeddings_adversarial, labels_pred_adver,
                                       layer_embeddings_noisy=layer_embeddings_noisy,
                                       labels_pred_noisy=labels_pred_noisy)

    def score(self, layer_embeddings, cleanup=True):
        """
        Same inputs and output as the `score` method of the class `DetectorLID`.
        """
        n_test = layer_embeddings[0].shape[0]
        labels_pred = self._gen_random_labels(n_test)

        return self.det_class_cond.score(layer_embeddings, labels_pred, cleanup=cleanup)


class DetectorLIDClassCond:
    _name = 'lid_class_cond'
    def __init__(self,
                 neighborhood_constant=NEIGHBORHOOD_CONST, n_neighbors=None,
                 metric='euclidean', metric_kwargs=None,
                 n_cv_folds=CROSS_VAL_SIZE,
                 c_search_values=None,
                 approx_nearest_neighbors=True,
                 skip_dim_reduction=True,
                 model_dim_reduction=None,
                 n_jobs=1,
                 max_iter=200,
                 balanced_classification=True,
                 low_memory=False,
                 save_knn_indices_to_file=True,
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
        :param n_cv_folds: number of cross-validation folds.
        :param c_search_values: list or array of search values for the logistic regression hyper-parameter `C`. The
                                default value is `None`.
        :param approx_nearest_neighbors: Set to True in order to use an approximate nearest neighbor algorithm to
                                         find the nearest neighbors. The NN-descent method is used for approximate
                                         nearest neighbor searches.
        :param skip_dim_reduction: Set to True in order to skip dimension reduction of the layer embeddings.
        :param model_dim_reduction: 1. None if dimension reduction is not required; (OR)
                                    2. Path to a file containing the saved dimension reduction model. This will be
                                       a pickle file that loads into a list of model dictionaries; (OR)
                                    3. The dimension reduction model loaded into memory from the pickle file.
        :param n_jobs: Number of parallel jobs or processes. Set to -1 to use all the available cpu cores.
        :param max_iter: Maximum number of iterations for the optimization of the logistic classifier. The default
                         value set by the scikit-learn library is 100, but sometimes this does not allow for
                         convergence. Hence, increasing it to 200 here.
        :param balanced_classification: Set to True to assign sample weights to balance the binary classification
                                        problem separating adversarial from non-adversarial samples.
        :param low_memory: Set to True to enable the low memory option of the `NN-descent` method. Note that this
                           is likely to increase the running time.
        :param save_knn_indices_to_file: Set to True in order to save the KNN indices from each layer and from each
                                         class to a pickle file to reduce memory usage. This may not be needed when
                                         the data size and/or the number of layers is small. It avoids potential
                                         out-of-memory errors at the expense of time taken to write and read the files.
        :param seed_rng: int value specifying the seed for the random number generator. This is passed around to
                         all the classes/functions that require random number generation. Set this to a fixed value
                         for reproducible results.
        """
        self.neighborhood_constant = neighborhood_constant
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.n_cv_folds = n_cv_folds
        self.c_search_values = c_search_values
        self.approx_nearest_neighbors = approx_nearest_neighbors
        self.skip_dim_reduction = skip_dim_reduction
        self.n_jobs = get_num_jobs(n_jobs)
        self.max_iter = max_iter
        self.balanced_classification = balanced_classification
        self.low_memory = low_memory
        self.save_knn_indices_to_file = save_knn_indices_to_file
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

        if self.c_search_values is None:
            # Default search values for the `C` parameter of logistic regression
            self.c_search_values = np.logspace(-4, 4, num=10)

        self.n_layers = None
        self.n_samples = []
        self.labels_unique = None
        # Index of train samples from each class based on the true class and predicted class labels
        self.indices_true = dict()
        self.indices_pred_normal = dict()
        self.indices_pred_adver = dict()
        self.indices_pred_noisy = dict()
        # Number of nearest neighbors for each class
        self.n_neighbors_per_class = dict()
        # KNN index
        self.index_knn = None
        # Logistic classification model for normal vs. adversarial
        self.model_logistic = None
        # Feature scaler
        self.scaler = None
        # Temporary directory to save the KNN index files
        self.temp_direc = None
        self.temp_knn_files = None

    def fit(self, layer_embeddings_normal, labels_normal, labels_pred_normal,
            layer_embeddings_adversarial, labels_pred_adversarial,
            layer_embeddings_noisy=None, labels_pred_noisy=None):
        """
        Extract the LID feature vector for normal, noisy, and adversarial samples and train a logistic classifier
        to separate adversarial samples from (normal + noisy). Cross-validation is used to select the hyper-parameter
        `C` using area under the ROC curve as the validation metric.

        NOTE:
        True labels and predicted labels are required for the normal feature embeddings.
        Only predicted labels are needed for the noisy and adversarial feature embeddings.

        :param layer_embeddings_normal: list of numpy arrays with the layer embeddings for normal samples.
                                        Length of the list is equal to the number of layers. The numpy array at
                                        index `i` has shape `(n, d_i)`, where `n` is the number of samples and `d_i`
                                        is the dimension of the embeddings at layer `i`.
        :param labels_normal: numpy array of class labels for the normal samples. Should have shape `(n, )`, where
                              `n` is the number of normal samples.
        :param labels_pred_normal: numpy array of DNN classifier predictions for the normal samples. Should have the
                                   same shape as `labels_normal`.
        :param layer_embeddings_adversarial: Same format as `layer_embeddings_normal`, but corresponding to
                                             the adversarial samples.
        :param labels_pred_adversarial: numpy array of DNN classifier predictions for the adversarial samples. Should
                                        have shape `(n, )`, where `n` is the number of adversarial samples.
        :param layer_embeddings_noisy: Same format as `layer_embeddings_normal`, but corresponding to the noisy
                                       samples. Can be set to `None` to exclude noisy data from training.
        :param labels_pred_noisy: numpy array of DNN classifier predictions for the noisy samples. Should have shape
                                  `(n, )`, where `n` is the number of noisy samples. Can be set to `None` to exclude
                                  noisy data from training.
        :return:
            (self, scores_normal, scores_adversarial) if layer_embeddings_noise is None
            (self, scores_normal, scores_adversarial, scores_noisy) otherwise.
            -------------------------------------------------------
            - self: trained instance of the class.
            - scores_normal: numpy array with the scores (decision function of the logistic classifier) for normal
                             samples. 1d array with the same number of samples as `layer_embeddings_normal`.
            - scores_noisy: scores corresponding to `layer_embeddings_noisy` if noisy training data is provided.
            - scores_adversarial: scores corresponding to `layer_embeddings_adversarial`.
        """
        self.n_layers = len(layer_embeddings_normal)
        logger.info("Number of layer embeddings: {:d}.".format(self.n_layers))
        if layer_embeddings_noisy is None:
            logger.info("Noisy training data not provided.")
            cond1 = False
            noisy_data = False
        else:
            cond1 = (len(layer_embeddings_noisy) != self.n_layers)
            noisy_data = True
            if labels_pred_noisy is None:
                raise ValueError("Class predictions are not provided for the noisy data")

        if cond1 or (len(layer_embeddings_adversarial) != self.n_layers):
            raise ValueError("The layer embeddings for noisy and attack samples must have the same length as that "
                             "of normal samples")

        if labels_normal.shape != labels_pred_normal.shape:
            raise ValueError("Length of arrays 'labels_normal' and 'labels_pred_normal' is not equal")

        # Number of samples in each of the categories
        self.n_samples = [
            layer_embeddings_normal[0].shape[0],
            layer_embeddings_noisy[0].shape[0] if noisy_data else 0,
            layer_embeddings_adversarial[0].shape[0]
        ]
        # Distinct class labels
        self.labels_unique = np.unique(labels_normal)
        for c in self.labels_unique:
            # Normal labeled samples from class `c`
            self.indices_true[c] = np.where(labels_normal == c)[0]
            # Normal samples predicted into class `c`
            self.indices_pred_normal[c] = np.where(labels_pred_normal == c)[0]
            # Adversarial samples predicted into class `c`
            self.indices_pred_adver[c] = np.where(labels_pred_adversarial == c)[0]
            if noisy_data:
                # Noisy samples predicted into class `c`
                self.indices_pred_noisy[c] = np.where(labels_pred_noisy == c)[0]

            # Number of nearest neighbors per class
            if self.n_neighbors is None:
                # Set based on the number of samples from this class and the neighborhood constant
                self.n_neighbors_per_class[c] = \
                    int(np.ceil(self.indices_true[c].shape[0] ** self.neighborhood_constant))
            else:
                # Use the value specified as input
                self.n_neighbors_per_class[c] = self.n_neighbors

        # The data arrays at all layers should have the same number of samples
        if not all([layer_embeddings_normal[i].shape[0] == self.n_samples[0] for i in range(self.n_layers)]):
            raise ValueError("Input 'layer_embeddings_normal' does not have the expected format")

        if noisy_data:
            if not all([layer_embeddings_noisy[i].shape[0] == self.n_samples[1] for i in range(self.n_layers)]):
                raise ValueError("Input 'layer_embeddings_noisy' does not have the expected format")

        if not all([layer_embeddings_adversarial[i].shape[0] == self.n_samples[2] for i in range(self.n_layers)]):
            raise ValueError("Input 'layer_embeddings_adversarial' does not have the expected format")

        if self.save_knn_indices_to_file:
            # Create a temporary directory for saving the KNN indices
            self.temp_direc = tempfile.mkdtemp(dir=os.getcwd())
            self.temp_knn_files = [''] * self.n_layers

        # KNN indices for the layer embeddings from each layer and each class
        self.index_knn = [dict() for _ in range(self.n_layers)]
        features_lid_normal = np.zeros((self.n_samples[0], self.n_layers))
        features_lid_noisy = np.zeros((self.n_samples[1], self.n_layers))
        features_lid_adversarial = np.zeros((self.n_samples[2], self.n_layers))
        for i in range(self.n_layers):
            logger.info("Processing layer {:d}:".format(i + 1))
            # Dimensionality reduction of the layer embeddings, if required
            if self.transform_models:
                data_normal = transform_data_from_model(layer_embeddings_normal[i], self.transform_models[i])
                data_adver = transform_data_from_model(layer_embeddings_adversarial[i], self.transform_models[i])
                if noisy_data:
                    data_noisy = transform_data_from_model(layer_embeddings_noisy[i], self.transform_models[i])
                else:
                    data_noisy = None

                d1 = layer_embeddings_normal[i].shape[1]
                d2 = data_normal.shape[1]
                if d2 < d1:
                    logger.info("Input dimension = {:d}, projected dimension = {:d}".format(d1, d2))
            else:
                data_normal = layer_embeddings_normal[i]
                data_adver = layer_embeddings_adversarial[i]
                if noisy_data:
                    data_noisy = layer_embeddings_noisy[i]
                else:
                    data_noisy = None

            for c in self.labels_unique:
                logger.info("Building a KNN index on the feature embeddings of normal samples from class {}".
                            format(c))
                self.index_knn[i][c] = KNNIndex(
                    data_normal[self.indices_true[c], :], n_neighbors=self.n_neighbors_per_class[c],
                    metric=self.metric, metric_kwargs=self.metric_kwargs,
                    approx_nearest_neighbors=self.approx_nearest_neighbors,
                    n_jobs=self.n_jobs,
                    low_memory=self.low_memory,
                    seed_rng=self.seed_rng
                )
                logger.info("Calculating LID estimates for the normal, noisy, and adversarial layer embeddings "
                            "predicted into class {}".format(c))
                # Distance to nearest neighbors of all labeled samples from class `c`
                _, nn_distances_temp = self.index_knn[i][c].query_self(k=self.n_neighbors_per_class[c])

                n_pred_normal = self.indices_pred_normal[c].shape[0]
                n_pred_adver = self.indices_pred_adver[c].shape[0]
                if noisy_data:
                    n_pred_noisy = self.indices_pred_noisy[c].shape[0]
                else:
                    n_pred_noisy = 0

                if n_pred_normal:
                    # Distance to nearest neighbors of samples predicted into class `c` that are also labeled as
                    # class `c`. These samples will be a part of the KNN index
                    nn_distances = helper_knn_distance(self.indices_pred_normal[c], self.indices_true[c],
                                                       nn_distances_temp)
                    mask = (nn_distances[:, 0] < 0.)
                    if np.any(mask):
                        # Distance to nearest neighbors of samples predicted into class `c` that are not labeled as
                        # class `c`. These samples will not be a part of the KNN index
                        ind_comp = self.indices_pred_normal[c][mask]
                        _, temp_arr = self.index_knn[i][c].query(data_normal[ind_comp, :],
                                                                 k=self.n_neighbors_per_class[c])
                        nn_distances[mask, :] = temp_arr

                    # LID estimates for the normal feature embeddings predicted into class `c`
                    features_lid_normal[self.indices_pred_normal[c], i] = lid_mle_amsaleg(nn_distances)

                # LID estimates for the noisy feature embeddings predicted into class `c`
                if n_pred_noisy:
                    temp_arr = data_noisy[self.indices_pred_noisy[c], :]
                    _, nn_distances = self.index_knn[i][c].query(temp_arr, k=self.n_neighbors_per_class[c])
                    features_lid_noisy[self.indices_pred_noisy[c], i] = lid_mle_amsaleg(nn_distances)

                # LID estimates for the adversarial feature embeddings predicted into class `c`
                if n_pred_adver:
                    temp_arr = data_adver[self.indices_pred_adver[c], :]
                    _, nn_distances = self.index_knn[i][c].query(temp_arr, k=self.n_neighbors_per_class[c])
                    features_lid_adversarial[self.indices_pred_adver[c], i] = lid_mle_amsaleg(nn_distances)

            if self.save_knn_indices_to_file:
                logger.info("Saving the KNN indices per class from layer {:d} to a pickle file".format(i + 1))
                self.temp_knn_files[i] = os.path.join(self.temp_direc, 'knn_indices_layer_{:d}.pkl'.format(i + 1))
                with open(self.temp_knn_files[i], 'wb') as fp:
                    pickle.dump(self.index_knn[i], fp)

                # Free up the allocated memory
                self.index_knn[i] = None

        # LID feature vectors and labels for the binary logistic classifier.
        # Normal and noisy samples are given label 0 and adversarial samples are given label 1
        n_pos = features_lid_adversarial.shape[0]
        if noisy_data:
            features_lid = np.concatenate([features_lid_normal, features_lid_noisy, features_lid_adversarial],
                                          axis=0)
            labels_bin = np.concatenate([np.zeros(features_lid_normal.shape[0], dtype=np.int),
                                         np.zeros(features_lid_noisy.shape[0], dtype=np.int),
                                         np.ones(n_pos, dtype=np.int)])
        else:
            features_lid = np.concatenate([features_lid_normal, features_lid_adversarial], axis=0)
            labels_bin = np.concatenate([np.zeros(features_lid_normal.shape[0], dtype=np.int),
                                         np.ones(n_pos, dtype=np.int)])

        pos_prop = n_pos / float(labels_bin.shape[0])
        # Randomly shuffle the samples to avoid determinism
        ind_perm = np.random.permutation(labels_bin.shape[0])
        features_lid = features_lid[ind_perm, :]
        labels_bin = labels_bin[ind_perm]
        # Min-max scaling for the LID features
        self.scaler = MinMaxScaler().fit(features_lid)
        features_lid = self.scaler.transform(features_lid)
        logger.info("Training a binary logistic classifier with {:d} samples and {:d} LID features.".
                    format(*features_lid.shape))
        logger.info("Using {:d}-fold cross-validation with area under ROC curve as the metric to select "
                    "the best regularization hyperparameter.".format(self.n_cv_folds))
        logger.info("Proportion of positive (adversarial or OOD) samples in the training data: {:.4f}".
                    format(pos_prop))
        class_weight = None
        if self.balanced_classification:
            if (pos_prop < 0.45) or (pos_prop > 0.55):
                class_weight = {0: 1.0 / (1 - pos_prop),
                                1: 1.0 / pos_prop}
                logger.info("Balancing the classes by assigning sample weight {:.4f} to class 0 and sample weight "
                            "{:.4f} to class 1".format(class_weight[0], class_weight[1]))

        self.model_logistic = LogisticRegressionCV(
            Cs=self.c_search_values,
            cv=self.n_cv_folds,
            penalty='l2',
            scoring='roc_auc',
            multi_class='auto',
            class_weight=class_weight,
            max_iter=self.max_iter,
            refit=True,
            n_jobs=self.n_jobs,
            random_state=self.seed_rng
        ).fit(features_lid, labels_bin)

        # Larger values of this score correspond to a higher probability of predicting class 1 (adversarial)
        scores_normal = self.model_logistic.decision_function(self.scaler.transform(features_lid_normal))
        scores_adversarial = self.model_logistic.decision_function(self.scaler.transform(features_lid_adversarial))
        if noisy_data:
            scores_noisy = self.model_logistic.decision_function(self.scaler.transform(features_lid_noisy))
            return self, scores_normal, scores_adversarial, scores_noisy
        else:
            return self, scores_normal, scores_adversarial

    def score(self, layer_embeddings, labels_pred, cleanup=True):
        """
        Given a list of layer embeddings for test samples, extract the layer-wise LID feature vector and return the
        decision function of the logistic classifier.
        :param layer_embeddings: list of numpy arrays with the layer embeddings for normal samples. Length of the
                                 list is equal to the number of layers. The numpy array at index `i` has shape
                                 `(n, d_i)`, where `n` is the number of samples and `d_i` is the dimension of the
                                 embeddings at layer `i`.
        :param labels_pred: numpy array with the predicted class labels for the samples in `layer_embeddings`.
        :param cleanup: If set to True, the temporary directory where the KNN index files are saved will be deleted
                        after scoring. If this method is to be called multiple times, set `cleanup = False` for all
                        calls except the last one.
        :return:
            - numpy array of detection scores for the test samples. Has shape `(n, )` where `n` is the number of
              samples. Larger values correspond to a higher confidence that the sample is adversarial.
        """
        n_test = layer_embeddings[0].shape[0]
        l = len(layer_embeddings)
        if l != self.n_layers:
            raise ValueError("Expecting {:d} layers in the input 'layer_embeddings', but received {:d} layers.".
                             format(self.n_layers, l))

        features_lid = np.zeros((n_test, self.n_layers))
        for i in range(self.n_layers):
            logger.info("Calculating LID features for layer {:d}".format(i + 1))
            if self.transform_models:
                # Dimension reduction
                data_proj = transform_data_from_model(layer_embeddings[i], self.transform_models[i])
            else:
                data_proj = layer_embeddings[i]

            if self.save_knn_indices_to_file:
                # logger.info("Loading the KNN indices per class from file")
                with open(self.temp_knn_files[i], 'rb') as fp:
                    self.index_knn[i] = pickle.load(fp)

            for c in self.labels_unique:
                ind = np.where(labels_pred == c)[0]
                if ind.shape[0]:
                    _, nn_distances = self.index_knn[i][c].query(data_proj[ind, :], k=self.n_neighbors_per_class[c])
                    features_lid[ind, i] = lid_mle_amsaleg(nn_distances)

            if self.save_knn_indices_to_file:
                # Free up the allocated memory
                self.index_knn[i] = None

        if cleanup and self.save_knn_indices_to_file:
            _ = subprocess.check_call(['rm', '-rf', self.temp_direc])

        features_lid = self.scaler.transform(features_lid)
        return self.model_logistic.decision_function(features_lid)
