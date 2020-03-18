"""
Implemention of the adversarial attack detection method from the paper:

Ma, Xingjun, et al. "Characterizing adversarial subspaces using local intrinsic dimensionality.",
 International conference on learning representations, 2018.
https://arxiv.org/pdf/1801.02613.pdf

Some of this code is borrowed from: https://github.com/xingjunm/lid_adversarial_subspace_detection

Note that this implementation does not use the mini-batching method to estimate LID as done in the paper.
Since the main utility of mini-batching was for computational efficiency, we instead use the approximate nearest
neighbors method for fast querying of neighbors from the full set of normal data.

"""
import numpy as np
import sys
import logging
from helpers.constants import (
    SEED_DEFAULT,
    NEIGHBORHOOD_CONST,
    CROSS_VAL_SIZE,
    METRIC_DEF
)
from helpers.knn_index import KNNIndex
from helpers.lid_estimators import lid_mle_amsaleg
from helpers.utils import get_num_jobs
from sklearn.linear_model import LogisticRegressionCV

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


class DetectorLID:
    def __init__(self,
                 neighborhood_constant=NEIGHBORHOOD_CONST, n_neighbors=None,
                 metric=METRIC_DEF, metric_kwargs=None,
                 n_cv_folds=CROSS_VAL_SIZE,
                 c_search_values=None,
                 approx_nearest_neighbors=True,
                 n_jobs=1,
                 max_iter=200,
                 balanced_classification=True,
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
        :param n_cv_folds: number of cross-validation folds.
        :param c_search_values: list or array of search values for the logistic regression hyper-parameter `C`. The
                                default value is `None`.
        :param approx_nearest_neighbors: Set to True in order to use an approximate nearest neighbor algorithm to
                                         find the nearest neighbors. The NN-descent method is used for approximate
                                         nearest neighbor searches.
        :param n_jobs: Number of parallel jobs or processes. Set to -1 to use all the available cpu cores.
        :param max_iter: Maximum number of iterations for the optimization of the logistic classifier. The default
                         value set by the scikit-learn library is 100, but sometimes this does not allow for
                         convergence. Hence, increasing it to 200 here.
        :param balanced_classification: Set to True to assign sample weights to balance the binary classification
                                        problem separating adversarial from non-adversarial samples.
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
        self.n_cv_folds = n_cv_folds
        self.c_search_values = c_search_values
        self.approx_nearest_neighbors = approx_nearest_neighbors
        self.n_jobs = get_num_jobs(n_jobs)
        self.max_iter = max_iter
        self.balanced_classification = balanced_classification
        self.low_memory = low_memory
        self.seed_rng = seed_rng

        np.random.seed(self.seed_rng)
        if self.c_search_values is None:
            # Default search values for the `C` parameter of logistic regression
            self.c_search_values = np.logspace(-4, 4, num=10)

        self.n_layers = None
        self.n_samples = []
        self.index_knn = None
        self.model_logistic = None

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
        else:
            cond1 = (len(layer_embeddings_noisy) != self.n_layers)

        if cond1 or (len(layer_embeddings_adversarial) != self.n_layers):
            raise ValueError("The layer embeddings for noisy and attack samples must have the same length as that "
                             "of normal samples")

        # Number of samples in each of the categories
        self.n_samples = [
            layer_embeddings_normal[0].shape[0],
            layer_embeddings_noisy[0].shape[0] if layer_embeddings_noisy else 0,
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

        if layer_embeddings_noisy:
            if not all([layer_embeddings_noisy[i].shape[0] == self.n_samples[1] for i in range(self.n_layers)]):
                raise ValueError("Input 'layer_embeddings_noisy' does not have the expected format")

        if not all([layer_embeddings_adversarial[i].shape[0] == self.n_samples[2] for i in range(self.n_layers)]):
            raise ValueError("Input 'layer_embeddings_adversarial' does not have the expected format")

        self.index_knn = [None for _ in range(self.n_layers)]
        features_lid_normal = np.zeros((self.n_samples[0], self.n_layers))
        features_lid_noisy = np.zeros((self.n_samples[1], self.n_layers))
        features_lid_adversarial = np.zeros((self.n_samples[2], self.n_layers))
        for i in range(self.n_layers):
            logger.info("Processing layer {:d}:".format(i + 1))
            logger.info("Building a KNN index on the feature embeddings of normal samples.")
            # Build a KNN index on the set of feature embeddings from normal samples from layer `i`
            self.index_knn[i] = KNNIndex(
                layer_embeddings_normal[i], n_neighbors=self.n_neighbors,
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

            if layer_embeddings_noisy:
                logger.info("Calculating LID estimates for the feature embeddings of noisy samples.")
                # Nearest neighbors of the noisy feature embeddings from this layer
                nn_indices, nn_distances = self.index_knn[i].query(layer_embeddings_noisy[i], k=self.n_neighbors)
                # LID estimates of the noisy feature embeddings from this layer
                features_lid_noisy[:, i] = lid_mle_amsaleg(nn_distances)

            logger.info("Calculating LID estimates for the feature embeddings of adversarial samples.")
            # Nearest neighbors of the adversarial feature embeddings from this layer
            nn_indices, nn_distances = self.index_knn[i].query(layer_embeddings_adversarial[i], k=self.n_neighbors)
            # LID estimates of the adversarial feature embeddings from this layer
            features_lid_adversarial[:, i] = lid_mle_amsaleg(nn_distances)

        # Feature vector and labels for the binary logistic classifier.
        # Normal and noisy samples are given labels 0 and adversarial samples are given label 1
        n_pos = features_lid_adversarial.shape[0]
        if layer_embeddings_noisy:
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
        logger.info("Training a binary logistic classifier with {:d} samples and {:d} LID features.".
                    format(*features_lid.shape))
        logger.info("Using {:d}-fold cross-validation with area under ROC curve as the metric to select "
                    "the best regularization hyperparameter.".format(self.n_cv_folds))
        logger.info("Proportion of positive (adversarial or OOD) samples in the training data: {:.4f}".
                    format(pos_prop))
        if self.balanced_classification:
            class_weight = {0: 1.0 / (1 - pos_prop),
                            1: 1.0 / pos_prop}
            logger.info("Balancing the classes by assigning sample weight {:.4f} to class 0 and sample weight "
                        "{:.4f} to class 1".format(class_weight[0], class_weight[1]))
        else:
            class_weight = None

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
        scores_normal = self.model_logistic.decision_function(features_lid_normal)
        scores_adversarial = self.model_logistic.decision_function(features_lid_adversarial)
        if layer_embeddings_noisy:
            scores_noisy = self.model_logistic.decision_function(features_lid_noisy)
            return self, scores_normal, scores_adversarial, scores_noisy
        else:
            return self, scores_normal, scores_adversarial

    def score(self, layer_embeddings):
        """
        Given a list of layer embeddings for test samples, extract the layer-wise LID feature vector and return the
        decision function of the logistic classifier.

        :param layer_embeddings: list of numpy arrays with the layer embeddings for normal samples. Length of the
                                 list is equal to the number of layers. The numpy array at index `i` has shape
                                 `(n, d_i)`, where `n` is the number of samples and `d_i` is the dimension of the
                                 embeddings at layer `i`.
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
            _, nn_distances = self.index_knn[i].query(layer_embeddings[i], k=self.n_neighbors)
            features_lid[:, i] = lid_mle_amsaleg(nn_distances)

        return self.model_logistic.decision_function(features_lid)
