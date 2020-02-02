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
import logging
from helpers.constants import (
    SEED_DEFAULT,
    NEIGHBORHOOD_CONST,
    CROSS_VAL_SIZE
)
from helpers.knn_index import KNNIndex
from helpers.lid_estimators import lid_mle_amsaleg
from helpers.utils import get_num_jobs
from sklearn.linear_model import LogisticRegressionCV

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STDEVS = {
    'mnist': {'fgsm': 0.271, 'bim-a': 0.111, 'bim-b': 0.167, 'cw-l2': 0.207},
    'cifar10': {'fgsm': 0.0504, 'bim-a': 0.0084, 'bim-b': 0.0428, 'cw-l2': 0.007},
    'svhn': {'fgsm': 0.133, 'bim-a': 0.0155, 'bim-b': 0.095, 'cw-l2': 0.008}
}
CLIP_MIN = -0.5
CLIP_MAX = 0.5


def flip(x, nb_diff):
    """
    Helper function for get_noisy_samples
    :param x:
    :param nb_diff:
    :return:
    """
    original_shape = x.shape
    x = np.copy(np.reshape(x, (-1,)))
    candidate_inds = np.where(x < CLIP_MAX)[0]
    #assert candidate_inds.shape[0] >= nb_diff
    if candidate_inds.shape[0] < nb_diff:
        return None

    inds = np.random.choice(candidate_inds, nb_diff)
    x[inds] = CLIP_MAX

    return np.reshape(x, original_shape)


def get_noisy_samples(X_test, X_test_adv, dataset, attack):
    """
    TODO
    :param X_test:
    :param X_test_adv:
    :param dataset:
    :param attack:
    :return:
    """
    if attack in ['jsma', 'cw-l0']:
        X_test_noisy = np.zeros_like(X_test)
        for i in range(len(X_test)):
            # Count the number of pixels that are different
            nb_diff = len(np.where(X_test[i] != X_test_adv[i])[0])
            # Randomly flip an equal number of pixels (flip means move to max
            # value of 1)
            X_test_noisy[i] = flip(X_test[i], nb_diff)
    else:
        warnings.warn("Important: using pre-set Gaussian scale sizes to craft noisy "
                      "samples. You will definitely need to manually tune the scale "
                      "according to the L2 print below, otherwise the result "
                      "will inaccurate. In future scale sizes will be inferred "
                      "automatically. For now, manually tune the scales around "
                      "mnist: L2/20.0, cifar: L2/54.0, svhn: L2/60.0")
        # Add Gaussian noise to the samples
        # print(STDEVS[dataset][attack])
        X_test_noisy = np.minimum(
            np.maximum(
                X_test + np.random.normal(loc=0, scale=STDEVS[dataset][attack],
                                          size=X_test.shape),
                CLIP_MIN
            ),
            CLIP_MAX
        )

    return X_test_noisy


class DetectorLID:
    def __init__(self,
                 neighborhood_constant=NEIGHBORHOOD_CONST, n_neighbors=None,
                 metric='euclidean', metric_kwargs=None,
                 n_cv_folds=CROSS_VAL_SIZE,
                 c_search_range=(-4, 4),
                 num_c_values=10,
                 approx_nearest_neighbors=True,
                 n_jobs=1,
                 low_memory=False,
                 seed_rng=SEED_DEFAULT):
        self.neighborhood_constant = neighborhood_constant
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.n_cv_folds = n_cv_folds
        self.c_search_range = c_search_range
        self.num_c_values = num_c_values
        self.approx_nearest_neighbors = approx_nearest_neighbors
        self.n_jobs = get_num_jobs(n_jobs)
        self.low_memory = low_memory
        self.seed_rng = seed_rng

        np.random.seed(self.seed_rng)
        self.n_layers = None
        self.n_samples = []
        self.index_knn = None
        self.model_logistic = None
        self.scores_train = None

    def fit(self, layer_embeddings_normal, layer_embeddings_noisy, layer_embeddings_adversarial):
        self.n_layers = len(layer_embeddings_normal)
        logger.info("Number of layer embeddings: {:d}.".format(self.n_layers))
        if (len(layer_embeddings_noisy) != self.n_layers) or (len(layer_embeddings_adversarial) != self.n_layers):
            raise ValueError("The layer embeddings for noisy and attack samples must have the same length as that "
                             "of normal samples")

        # Number of samples in each of the categories
        self.n_samples = [
            layer_embeddings_normal[0].shape[0],
            layer_embeddings_noisy[0].shape[0],
            layer_embeddings_adversarial[0].shape[0]
        ]
        # Number of nearest neighbors
        if self.n_neighbors is None:
            # Set number of nearest neighbors based on the data size (of normal samples) and the
            # neighborhood constant
            self.n_neighbors = int(np.ceil(self.n_train[0] ** self.neighborhood_constant))

        # The data arrays at all layers should have the same number of samples
        if not all([layer_embeddings_normal[i].shape[0] == self.n_samples[0] for i in range(self.n_layers)]):
            raise ValueError("Input 'layer_embeddings_normal' does not have the expected format")

        if not all([layer_embeddings_noisy[i].shape[0] == self.n_samples[1] for i in range(self.n_layers)]):
            raise ValueError("Input 'layer_embeddings_noisy' does not have the expected format")

        if not all([layer_embeddings_adversarial[i].shape[0] == self.n_samples[2] for i in range(self.n_layers)]):
            raise ValueError("Input 'layer_embeddings_adversarial' does not have the expected format")

        features_lid_normal = np.zeros((self.n_samples[0], self.n_layers))
        features_lid_noisy = np.zeros((self.n_samples[1], self.n_layers))
        features_lid_adversarial = np.zeros((self.n_samples[2], self.n_layers))
        for i in range(self.n_layers):
            logger.info("Processing layer {:d}:".format(i + 1))
            logger.info("Building a KNN index on the feature embeddings of normal samples.")
            # Build a KNN index on the set of feature embeddings from normal samples from layer `i`
            self.index_knn = KNNIndex(
                layer_embeddings_normal[i], n_neighbors=self.n_neighbors,
                metric=self.metric, metric_kwargs=self.metric_kwargs,
                approx_nearest_neighbors=self.approx_nearest_neighbors,
                n_jobs=self.n_jobs,
                low_memory=self.low_memory,
                seed_rng=self.seed_rng
            )
            logger.info("Calculating LID estimates for the feature embeddings of normal samples.")
            # Nearest neighbors of the normal feature embeddings from this layer
            nn_indices, nn_distances = self.index_knn.query_self(k=self.n_neighbors)
            # LID estimates of the normal feature embeddings from this layer
            features_lid_normal[:, i] = lid_mle_amsaleg(nn_distances)

            logger.info("Calculating LID estimates for the feature embeddings of noisy samples.")
            # Nearest neighbors of the noisy feature embeddings from this layer
            nn_indices, nn_distances = self.index_knn.query(layer_embeddings_noisy[i], k=self.n_neighbors)
            # LID estimates of the noisy feature embeddings from this layer
            features_lid_noisy[:, i] = lid_mle_amsaleg(nn_distances)

            logger.info("Calculating LID estimates for the feature embeddings of adversarial samples.")
            # Nearest neighbors of the adversarial feature embeddings from this layer
            nn_indices, nn_distances = self.index_knn.query(layer_embeddings_adversarial[i], k=self.n_neighbors)
            # LID estimates of the adversarial feature embeddings from this layer
            features_lid_adversarial[:, i] = lid_mle_amsaleg(nn_distances)

        # Feature vector and labels for the binary logistic classifier
        features_lid = np.concatenate([features_lid_normal, features_lid_noisy, features_lid_adversarial], axis=0)
        # Normal and noisy samples are given labels 0 and adversarial samples are given label 1
        labels = np.concatenate([np.zeros(features_lid_normal.shape[0], dtype=np.int),
                                 np.zeros(features_lid_noisy.shape[0], dtype=np.int),
                                 np.ones(features_lid_adversarial.shape[0], dtype=np.int)])
        ns = labels.shape[0]
        # Randomly shuffle the samples to avoid determinism
        ind_perm = np.random.permutation(ns)
        features_lid = features_lid[ind_perm, :]
        labels = labels[ind_perm]

        logger.info("Training a binary logistic classifier with {:d} samples and {:d} LID features.".
                    format(*features_lid.shape))
        logger.info("Using {:d}-fold cross-validation with area under ROC curve (AUC) as the metric to select "
                    "the best C hyper-parameter.".format(self.n_cv_folds))

        Cs = np.logspace(self.c_search_range[0], self.c_search_range[1], num=self.num_c_values)
        self.model_logistic = LogisticRegressionCV(
            Cs=Cs,
            cv=self.n_cv_folds,
            penalty='l2',
            scoring='roc_auc',
            refit=True,
            n_jobs=self.n_jobs,
            random_state=self.seed_rng
        ).fit(features_lid, labels)

        # Larger values of this score correspond to a higher probability of predicting class 1 (adversarial)
        self.scores_train = self.model_logistic.decision_function(features_lid)
        return self

    def score(self, layer_embeddings):
        n_test = layer_embeddings[0].shape[0]
        l = len(layer_embeddings)
        if l != self.n_layers:
            raise ValueError("Expecting {:d} layers in the input 'layer_embeddings', but received {:d} layers.".
                             format(self.n_layers, l))

        features_lid = np.zeros((n_test, self.n_layers))
        for i in range(self.n_layers):
            logger.info("Calculating LID features for layer {:d}:".format(i + 1))
            nn_indices, nn_distances = self.index_knn.query(layer_embeddings[i], k=self.n_neighbors)
            features_lid[:, i] = lid_mle_amsaleg(nn_distances)

        return self.model_logistic.decision_function(features_lid)
