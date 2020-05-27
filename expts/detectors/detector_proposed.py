"""
Implemention of the proposed adversarial and OOD detection methods.
"""
import numpy as np
import sys
import torch
import logging
import copy
from helpers.constants import (
    SEED_DEFAULT,
    NEIGHBORHOOD_CONST,
    METRIC_DEF,
    NUM_TOP_RANKED,
    TEST_STATS_SUPPORTED,
    SCORE_TYPES,
    NUM_RANDOM_SAMPLES
)
from helpers.dimension_reduction_methods import (
    transform_data_from_model,
    load_dimension_reduction_models
)
from helpers.utils import (
    log_sum_exp,
    combine_and_vectorize,
    extract_layer_embeddings
)
from detectors.pvalue_estimation import (
    pvalue_score,
    pvalue_score_all_pairs
)
from detectors.test_statistics_layers import (
    MultinomialScore,
    BinomialScore,
    LIDScore,
    LLEScore,
    DistanceScore,
    TrustScore
)
from helpers.density_model_layer_statistics import (
    train_log_normal_mixture,
    score_log_normal_mixture
)
from detectors.localized_pvalue_estimation import averaged_KLPE_anomaly_detection

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def transform_layer_embeddings(embeddings_in, transform_models):
    """
    Perform dimension reduction on the data embeddings from each layer. The transformation or projection matrix
    for each layer is provided via the input `transform_models`.

    NOTE: In order to skip dimension reduction at a particular layer, the corresponding element of
    `transform_models` can be set to `None`. Thus, a list of `None` values can be passed to completely skip
    dimension reduction.

    :param embeddings_in: list of data embeddings per layer. `embeddings_in[i]` is a list of numpy arrays
                          corresponding to the data batches from layer `i`.
    :param transform_models: A list of dictionaries with the transformation models per layer. The length of
                             `transform_models` should be equal to the length of `embeddings_in`.
    :return: list of transformed data arrays, one per layer.
    """
    n_layers = len(embeddings_in)
    assert len(transform_models) == n_layers, ("Length of 'transform_models' is not equal to the length of "
                                               "'embeddings_in'")
    embeddings_out = []
    for i in range(n_layers):
        logger.info("Transforming the embeddings from layer {:d}:".format(i + 1))
        embeddings_out.append(transform_data_from_model(embeddings_in[i], transform_models[i]))
        logger.info("Input dimension = {:d}, projected dimension = {:d}".format(embeddings_in[i].shape[1],
                                                                                embeddings_out[-1].shape[1]))

    return embeddings_out


class DetectorLayerStatistics:
    """
    Main class implementing the adversarial and OOD detector based on joint hypothesis testing of test statistics
    calculated at the layers of a trained DNN.

    Harmonic mean method for combining p-values from multiple tetsts:
        - https://en.wikipedia.org/wiki/Harmonic_mean_p-value
        - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6347718/

    Fisher's method for combining p-values from multiple tests:
        https://en.wikipedia.org/wiki/Fisher%27s_method
    """
    _name = 'proposed'
    def __init__(self,
                 layer_statistic='multinomial',
                 score_type='pvalue',
                 ood_detection=False,
                 pvalue_fusion='fisher',
                 use_top_ranked=False,
                 num_top_ranked=NUM_TOP_RANKED,
                 skip_dim_reduction=False,
                 model_dim_reduction=None,
                 neighborhood_constant=NEIGHBORHOOD_CONST, n_neighbors=None,
                 metric=METRIC_DEF, metric_kwargs=None,
                 approx_nearest_neighbors=True,
                 n_jobs=1,
                 low_memory=False,
                 seed_rng=SEED_DEFAULT):
        """

        :param layer_statistic: Type of test statistic to calculate at the layers. Valid values are 'multinomial',
                                'binomial', 'lid', and 'lle'.
        :param score_type: Name of the scoring method to use. Valid options are: 'density' and 'pvalue'.
        :param ood_detection: Set to True to perform out-of-distribution detection instead of adversarial detection.
        :param pvalue_fusion: Method for combining the p-values across the layers. Options are 'harmonic_mean'
                              and 'fisher'. The former corresponds to the weighted harmonic mean of the p-values
                              and the latter corresponds to Fisher's method of combining p-values. This input is
                              used only when `score_type = 'pvalue'`.
        :param use_top_ranked: Set to True in order to use only a few top ranked test statistics for detection.
        :param num_top_ranked: If `use_top_ranked` is set to True, this specifies the number of top-ranked test
                               statistics to use for detection. This number should be smaller than the number of
                               layers considered for detection.
        :param skip_dim_reduction: Set to True in order to skip dimension reduction of the layer embeddings.
        :param model_dim_reduction: 1. None if dimension reduction is not required; (OR)
                                    2. Path to a file containing the saved dimension reduction model. This will be
                                       a pickle file that loads into a list of model dictionaries; (OR)
                                    3. The dimension reduction model loaded into memory from the pickle file.
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
        :param n_jobs: Number of parallel jobs or processes. Set to -1 to use all the available cpu cores.
        :param low_memory: Set to True to enable the low memory option of the `NN-descent` method. Note that this
                           is likely to increase the running time.
        :param seed_rng: int value specifying the seed for the random number generator. This is passed around to
                         all the classes/functions that require random number generation. Set this to a fixed value
                         for reproducible results.
        """
        self.layer_statistic = layer_statistic.lower()
        self.score_type = score_type.lower()
        self.ood_detection = ood_detection
        self.pvalue_fusion = pvalue_fusion
        self.use_top_ranked = use_top_ranked
        self.num_top_ranked = num_top_ranked
        self.skip_dim_reduction = skip_dim_reduction
        self.neighborhood_constant = neighborhood_constant
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.approx_nearest_neighbors = approx_nearest_neighbors
        self.n_jobs = n_jobs
        self.low_memory = low_memory
        self.seed_rng = seed_rng

        np.random.seed(self.seed_rng)
        if self.layer_statistic not in TEST_STATS_SUPPORTED:
            raise ValueError("Invalid value '{}' for the input argument 'layer_statistic'.".
                             format(self.layer_statistic))

        if self.score_type not in SCORE_TYPES:
            raise ValueError("Invalid value '{}' for the input argument 'score_type'.".format(self.score_type))

        if self.pvalue_fusion not in ['harmonic_mean', 'fisher']:
            raise ValueError("Invalid value '{}' for the input argument 'pvalue_fusion'.".format(self.pvalue_fusion))

        if self.layer_statistic in {'lid', 'lle'}:
            if not self.skip_dim_reduction:
                logger.warning("Option 'skip_dim_reduction' is set to False for the test statistic '{}'. Making sure "
                               "that this is the intended setting.".format(self.layer_statistic))

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
        # List of test statistic model instances for each layer
        self.test_stats_models = []
        # dict mapping each class `c` to the joint density model of the test statistics conditioned on the predicted
        # or true class being `c`
        self.density_models_pred = dict()
        self.density_models_true = dict()
        # Negative log density values of data randomly sampled from the joint density models of the test statistics
        self.samples_neg_log_dens_pred = dict()
        self.samples_neg_log_dens_true = dict()
        # Log of the class prior probabilities estimated from the training data labels
        self.log_class_priors = None
        # Localized p-value estimation models for the data conditioned on each predicted and true class
        self.klpe_models_pred = dict()
        self.klpe_models_true = dict()
        # Test statistics calculated on the training data passed to the `fit` method. These test statistics follow
        # the distribution under the null hypothesis of no adversarial or OOD data
        self.test_stats_pred_null = None
        self.test_stats_true_null = None

    def fit(self, layer_embeddings, labels, labels_pred, **kwargs):
        """
        Estimate parameters of the detection method given natural (non-adversarial) input data.
        NOTE: Inputs to this method can be obtained by calling the function `extract_layer_embeddings`.

        :param layer_embeddings: list of numpy arrays with the layer embedding data. Length of the list is equal to
                                 the number of layers. The numpy array at index `i` has shape `(n, d_i)`, where `n`
                                 is the number of samples and `d_i` is the dimension of the embeddings at layer `i`.
        :param labels: numpy array of labels for the classification problem addressed by the DNN. Should have shape
                       `(n, )`, where `n` is the number of samples.
        :param labels_pred: numpy array of class predictions made by the DNN. Should have the same shape as `labels`.
        :param kwargs: dict with additional keyword arguments that can be passed to the `fit` method of the test
                       statistic class.

        :return: Instance of the class with all parameters fit to the data.
        """
        self.n_layers = len(layer_embeddings)
        self.labels_unique = np.unique(labels)
        self.n_classes = len(self.labels_unique)
        self.n_samples = labels.shape[0]

        logger.info("Number of classes: {:d}.".format(self.n_classes))
        logger.info("Number of layer embeddings: {:d}.".format(self.n_layers))
        logger.info("Number of samples: {:d}.".format(self.n_samples))
        logger.info("Test statistic calculated at each layer: {}.".format(self.layer_statistic))
        if labels_pred.shape[0] != self.n_samples:
            raise ValueError("Inputs 'labels' and 'labels_pred' do not have the same size.")

        if not all([layer_embeddings[i].shape[0] == self.n_samples for i in range(self.n_layers)]):
            raise ValueError("Input 'layer_embeddings' does not have the expected format")

        if self.use_top_ranked:
            if self.num_top_ranked > self.n_layers:
                logger.warning("Number of top-ranked layer statistics cannot be larger than the number of layers. "
                               "Setting it equal to the number of layers ({:d}).".format(self.n_layers))
                self.num_top_ranked = self.n_layers

        self.log_class_priors = np.zeros(self.n_classes)
        indices_true = dict()
        indices_pred = dict()
        test_stats_true = dict()
        pvalues_true = dict()
        test_stats_pred = dict()
        pvalues_pred = dict()
        for c in self.labels_unique:
            indices_true[c] = np.where(labels == c)[0]
            indices_pred[c] = np.where(labels_pred == c)[0]
            # Test statistics and negative log p-values across the layers for the samples labeled into class `c`
            test_stats_true[c] = np.zeros((indices_true[c].shape[0], self.n_layers))
            pvalues_true[c] = np.zeros((indices_true[c].shape[0], self.n_layers))

            # Test statistics and negative log p-values across the layers for the samples predicted into class `c`
            test_stats_pred[c] = np.zeros((indices_pred[c].shape[0], self.n_layers))
            pvalues_pred[c] = np.zeros((indices_pred[c].shape[0], self.n_layers))

            # Log of the class prior probability
            self.log_class_priors[c] = indices_true[c].shape[0]

        self.log_class_priors = np.log(self.log_class_priors) - np.log(self.n_samples)

        for i in range(self.n_layers):
            if self.transform_models:
                logger.info("Transforming the embeddings from layer {:d}.".format(i + 1))
                data_proj = transform_data_from_model(layer_embeddings[i], self.transform_models[i])
                logger.info("Input dimension = {:d}, projected dimension = {:d}".
                            format(layer_embeddings[i].shape[1], data_proj.shape[1]))
            else:
                data_proj = layer_embeddings[i]

            logger.info("Parameter estimation and test statistics calculation for layer {:d}:".format(i + 1))
            ts_obj = None
            # Bootstrap p-values are used only if `self.use_top_ranked = True` because in this case the test
            # statistics across the layers are ranked based on the p-values
            kwargs_fit = {'bootstrap': self.use_top_ranked}
            if self.layer_statistic == 'multinomial':
                ts_obj = MultinomialScore(
                    neighborhood_constant=self.neighborhood_constant,
                    n_neighbors=self.n_neighbors,
                    metric=self.metric,
                    metric_kwargs=self.metric_kwargs,
                    shared_nearest_neighbors=False,
                    approx_nearest_neighbors=self.approx_nearest_neighbors,
                    n_jobs=self.n_jobs,
                    low_memory=self.low_memory,
                    seed_rng=self.seed_rng
                )
                if 'combine_low_proba_classes' in kwargs:
                    kwargs_fit['combine_low_proba_classes'] = kwargs['combine_low_proba_classes']
                if 'n_classes_multinom' in kwargs:
                    kwargs_fit['n_classes_multinom'] = kwargs['n_classes_multinom']

            elif self.layer_statistic == 'binomial':
                ts_obj = BinomialScore(
                    neighborhood_constant=self.neighborhood_constant,
                    n_neighbors=self.n_neighbors,
                    metric=self.metric,
                    metric_kwargs=self.metric_kwargs,
                    shared_nearest_neighbors=False,
                    approx_nearest_neighbors=self.approx_nearest_neighbors,
                    n_jobs=self.n_jobs,
                    low_memory=self.low_memory,
                    seed_rng=self.seed_rng
                )
            elif self.layer_statistic == 'lid':
                ts_obj = LIDScore(
                    neighborhood_constant=self.neighborhood_constant,
                    n_neighbors=self.n_neighbors,
                    metric='euclidean',     # use 'euclidean' metric for LID estimation
                    metric_kwargs=None,
                    approx_nearest_neighbors=self.approx_nearest_neighbors,
                    n_jobs=self.n_jobs,
                    low_memory=self.low_memory,
                    seed_rng=self.seed_rng
                )
            elif self.layer_statistic == 'lle':
                ts_obj = LLEScore(
                    neighborhood_constant=self.neighborhood_constant,
                    n_neighbors=self.n_neighbors,
                    metric=self.metric,
                    metric_kwargs=self.metric_kwargs,
                    approx_nearest_neighbors=self.approx_nearest_neighbors,
                    n_jobs=self.n_jobs,
                    low_memory=self.low_memory,
                    seed_rng=self.seed_rng
                )
            elif self.layer_statistic == 'distance':
                ts_obj = DistanceScore(
                    neighborhood_constant=self.neighborhood_constant,
                    n_neighbors=self.n_neighbors,
                    metric=self.metric,
                    metric_kwargs=self.metric_kwargs,
                    approx_nearest_neighbors=self.approx_nearest_neighbors,
                    n_jobs=self.n_jobs,
                    low_memory=self.low_memory,
                    seed_rng=self.seed_rng
                )
            elif self.layer_statistic == 'trust':
                ts_obj = TrustScore(
                    neighborhood_constant=self.neighborhood_constant,
                    n_neighbors=self.n_neighbors,
                    metric=self.metric,
                    metric_kwargs=self.metric_kwargs,
                    approx_nearest_neighbors=self.approx_nearest_neighbors,
                    n_jobs=self.n_jobs,
                    low_memory=self.low_memory,
                    seed_rng=self.seed_rng
                )

            test_stats_temp, pvalues_temp = ts_obj.fit(
                data_proj, labels, labels_pred, labels_unique=self.labels_unique, **kwargs_fit
            )
            '''
            - `test_stats_temp` will be a numpy array of shape `(self.n_samples, self.n_classes + 1)` with a vector 
            of test statistics for each sample.
            The first column `test_stats_temp[:, 0]` gives the scores conditioned on the predicted class.
            The remaining columns `test_stats_temp[:, i]` for `i = 1, 2, . . .` gives the scores conditioned on 
            `i - 1` being the candidate true class for the sample.
            - `pvalues_temp` is also a numpy array of the same shape with the negative log transformed p-values 
            corresponding to the test statistics.
            '''
            self.test_stats_models.append(ts_obj)
            for j, c in enumerate(self.labels_unique):
                # Test statistics and negative log p-values from layer `i`
                test_stats_pred[c][:, i] = test_stats_temp[indices_pred[c], 0]
                pvalues_pred[c][:, i] = pvalues_temp[indices_pred[c], 0]
                test_stats_true[c][:, i] = test_stats_temp[indices_true[c], j + 1]
                pvalues_true[c][:, i] = pvalues_temp[indices_true[c], j + 1]

        for c in self.labels_unique:
            if self.use_top_ranked:
                logger.info("Using the test statistics corresponding to the smallest (largest) {:d} p-values "
                            "conditioned on the predicted (true) class.".format(self.num_top_ranked))
                # For the test statistics conditioned on the predicted class, take the largest
                # `self.num_top_ranked` negative log-transformed p-values across the layers
                test_stats_pred[c], pvalues_pred[c] = self._get_top_ranked(
                    test_stats_pred[c], pvalues_pred[c], reverse=True
                )
                # For the test statistics conditioned on the true class, take the smallest `self.num_top_ranked`
                # negative log-transformed p-values across the layers
                test_stats_true[c], pvalues_true[c] = self._get_top_ranked(test_stats_true[c], pvalues_true[c])

            if self.score_type == 'density':
                logger.info("Learning a joint probability density model for the test statistics conditioned on the "
                            "predicted class '{}':".format(c))
                logger.info("Number of samples = {:d}, dimension = {:d}".format(*test_stats_pred[c].shape))
                self.density_models_pred[c] = train_log_normal_mixture(test_stats_pred[c], seed_rng=self.seed_rng)

                # Negative log density of the data used to fit the model
                arr1 = -1. * score_log_normal_mixture(test_stats_pred[c], self.density_models_pred[c],
                                                      log_transform=True)
                # Generate a large number of random samples from the model
                test_stats_rand_sample, _ = self.density_models_pred[c].sample(n_samples=NUM_RANDOM_SAMPLES)
                # Negative log density of the generated random samples. Log transformation is not needed since the
                # samples are generated from the model
                arr2 = -1. * score_log_normal_mixture(test_stats_rand_sample, self.density_models_pred[c],
                                                      log_transform=False)
                self.samples_neg_log_dens_pred[c] = np.concatenate([arr1, arr2])
                logger.info("Number of log-density sample values used for estimating p-values: {:d}".
                            format(self.samples_neg_log_dens_pred[c].shape[0]))

                logger.info("Learning a joint probability density model for the test statistics conditioned on the "
                            "true class '{}':".format(c))
                logger.info("Number of samples = {:d}, dimension = {:d}".format(*test_stats_true[c].shape))
                self.density_models_true[c] = train_log_normal_mixture(test_stats_true[c], seed_rng=self.seed_rng)

                # Negative log density of the data used to fit the model
                arr1 = -1. * score_log_normal_mixture(test_stats_true[c], self.density_models_true[c],
                                                      log_transform=True)
                # Generate a large number of random samples from the model
                test_stats_rand_sample, _ = self.density_models_true[c].sample(n_samples=NUM_RANDOM_SAMPLES)
                # Negative log density of the generated random samples
                arr2 = -1. * score_log_normal_mixture(test_stats_rand_sample, self.density_models_true[c],
                                                      log_transform=False)
                self.samples_neg_log_dens_true[c] = np.concatenate([arr1, arr2])
                logger.info("Number of log-density sample values used for estimating p-values: {:d}".
                            format(self.samples_neg_log_dens_true[c].shape[0]))

            if self.score_type == 'klpe':
                # Not setting the number of neighbors here. This will be automatically set based on the number of
                # samples per class
                kwargs_lpe = {
                    'neighborhood_constant': self.neighborhood_constant,
                    'metric': self.metric,
                    'metric_kwargs': self.metric_kwargs,
                    'approx_nearest_neighbors': self.approx_nearest_neighbors,
                    'n_jobs': self.n_jobs,
                    'seed_rng': self.seed_rng
                }
                logger.info("Fitting the localized p-value estimation model for the test statistics conditioned on "
                            "the predicted class {}:".format(c))
                self.klpe_models_pred[c] = averaged_KLPE_anomaly_detection(**kwargs_lpe)
                self.klpe_models_pred[c].fit(test_stats_pred[c])

                logger.info("Fitting the localized p-value estimation model for the test statistics conditioned on "
                            "the true class {}:".format(c))
                self.klpe_models_true[c] = averaged_KLPE_anomaly_detection(**kwargs_lpe)
                self.klpe_models_true[c].fit(test_stats_true[c])

        self.test_stats_pred_null = test_stats_pred
        self.test_stats_true_null = test_stats_true
        return self

    def score(self, layer_embeddings, labels_pred, return_corrected_predictions=False, start_layer=0,
              test_layer_pairs=True, is_train=False):
        """
        Given the layer embeddings (including possibly the input itself) and the predicted classes for test data,
        score them on how likely they are to be adversarial or out-of-distribution (OOD). Larger values of the
        scores correspond to a higher probability that the test sample is adversarial or OOD. The scores can be
        thresholded, with values above the threshold declared as adversarial or OOD. The threshold can be set such
        that the detector has a target false positive rate.

        :param layer_embeddings: list of numpy arrays with the layer embedding data. Length of the list is equal to
                                 the number of layers. The numpy array at index `i` has shape `(n, d_i)`, where `n`
                                 is the number of samples and `d_i` is the dimension of the embeddings at layer `i`.
        :param labels_pred: numpy array of class predictions made by the DNN.
        :param return_corrected_predictions: Set to True in order to get the most probable class prediction based
                                             on Bayes class posterior given the test statistic vector. Note that this
                                             will change the returned values.
        :param start_layer: Starting index of the layers to include in the p-value fusion. Set to 0 to include all
                            the layers. Set to negative values such as -1, -2, -3 using the same convention as
                            python indexing. For example, a value of `-3` implies the last 3 layers are included.
        :param test_layer_pairs: Set to True in order to estimate p-values for test statistics from all pairs of
                                 layers. These additional p-values are used by the method which combines p-values
                                 using Fisher's method, harmonic mean of p-values etc.
        :param is_train: Set to True if the inputs are the same non-adversarial inputs used with the `fit` method.

        :return: (scores [, corrected_classes])
            - scores: numpy array of scores for detection or ranking. The array should have shape
                      `(labels_pred.shape[0], )` and larger values correspond to a higher higher probability that
                      the sample is adversarial or OOD. Score corresponding to OOD detection is returned if
                      `self.ood_detection = True`.
            # returned only if `return_corrected_predictions = True`
            - corrected_classes: numpy array of the corrected class predictions. Has same shape and dtype as the
                                 array `labels_pred`.
        """
        n_test = labels_pred.shape[0]
        l = len(layer_embeddings)
        if l != self.n_layers:
            raise ValueError("Expecting {:d} layers in the input data, but received {:d}".format(self.n_layers, l))

        # Should bootstrap resampling be used to estimate the p-values at each layer?
        bootstrap = True
        if self.score_type in ('density', 'klpe'):
            if not self.use_top_ranked:
                # The p-values estimated are never used in this case. Therefore, skipping bootstrap to make it faster
                bootstrap = False

        # Test statistics at each layer conditioned on the predicted class and candidate true classes
        test_stats_pred = np.zeros((n_test, self.n_layers))
        pvalues_pred = np.zeros((n_test, self.n_layers))
        test_stats_true = {c: np.zeros((n_test, self.n_layers)) for c in self.labels_unique}
        pvalues_true = {c: np.zeros((n_test, self.n_layers)) for c in self.labels_unique}
        for i in range(self.n_layers):
            if self.transform_models:
                # Dimension reduction
                data_proj = transform_data_from_model(layer_embeddings[i], self.transform_models[i])
            else:
                data_proj = layer_embeddings[i]

            # Test statistics and negative log p-values for layer `i`
            test_stats_temp, pvalues_temp = self.test_stats_models[i].score(data_proj, labels_pred, is_train=is_train,
                                                                            bootstrap=bootstrap)
            # `test_stats_temp` and `pvalues_temp` will have shape `(n_test, self.n_classes + 1)`

            test_stats_pred[:, i] = test_stats_temp[:, 0]
            pvalues_pred[:, i] = pvalues_temp[:, 0]
            for j, c in enumerate(self.labels_unique):
                test_stats_true[c][:, i] = test_stats_temp[:, j + 1]
                pvalues_true[c][:, i] = pvalues_temp[:, j + 1]

        if self.use_top_ranked:
            # For the test statistics conditioned on the predicted class, take the largest `self.num_top_ranked`
            # negative log p-values across the layers
            test_stats_pred, pvalues_pred = self._get_top_ranked(test_stats_pred, pvalues_pred, reverse=True)

            # For the test statistics conditioned on the true class, take the smallest `self.num_top_ranked`
            # negative log p-values across the layers
            for c in self.labels_unique:
                test_stats_true[c], pvalues_true[c] = self._get_top_ranked(test_stats_true[c], pvalues_true[c])

        # Adversarial or OOD scores for the test samples and the corrected class predictions
        if self.score_type == 'density':
            scores_adver, scores_ood, corrected_classes = self._score_density_based(
                labels_pred, test_stats_pred, test_stats_true,
                return_corrected_predictions=return_corrected_predictions
            )
        elif self.score_type == 'pvalue':
            if test_layer_pairs:
                n_pairs = int(0.5 * self.n_layers * (self.n_layers - 1))
                # logger.info("Estimating p-values for the test statistics from {:d} layer pairs.".format(n_pairs))
                pvalues_pred_pairs = np.zeros((n_test, n_pairs))
                pvalues_true_pairs = dict()
                for c in self.labels_unique:
                    # Samples predicted into class `c`
                    ind = np.where(labels_pred == c)[0]
                    pvalues_pred_pairs[ind, :] = pvalue_score_all_pairs(
                        self.test_stats_pred_null[c], test_stats_pred[ind, :], log_transform=True, bootstrap=bootstrap
                    )
                    pvalues_true_pairs[c] = pvalue_score_all_pairs(
                        self.test_stats_true_null[c], test_stats_true[c], log_transform=True, bootstrap=bootstrap
                    )
                    # Append columns corresponding to the p-values from the layer pairs
                    pvalues_true[c] = np.hstack((pvalues_true[c], pvalues_true_pairs[c]))

                # Append columns corresponding to the p-values from the layer pairs
                pvalues_pred = np.hstack((pvalues_pred, pvalues_pred_pairs))

            scores_adver, scores_ood, corrected_classes = self._score_pvalue_based(
                labels_pred, pvalues_pred, pvalues_true,
                return_corrected_predictions=return_corrected_predictions, start_layer=start_layer
            )
        elif self.score_type == 'klpe':
            scores_adver, scores_ood, corrected_classes = self._score_klpe(
                labels_pred, test_stats_pred, test_stats_true,
                return_corrected_predictions=return_corrected_predictions
            )
        else:
            raise ValueError("Invalid score type '{}'".format(self.score_type))

        if return_corrected_predictions:
            if self.ood_detection:
                return scores_ood, corrected_classes
            else:
                return scores_adver, corrected_classes
        else:
            if self.ood_detection:
                return scores_ood
            else:
                return scores_adver


    def _score_density_based(self, labels_pred, test_stats_pred, test_stats_true,
                             return_corrected_predictions=False):
        """
        Scoring method based on modeling the joint probability density of the test statistics, conditioned on the
        predicted and true class.

        :param labels_pred: Same as the method `score`.
        :param test_stats_pred: numpy array with the test statistics from the different layers, conditioned on the
                                predicted class of the test samples. Is a numpy array of shape `(n_test, n_layers)`,
                                where `n_test` and `n_layers` are the number of test samples and number of layers
                                respectively.
        :param test_stats_true: dict with the test statistics from the different layers, conditioned on each candidate
                                true class (since this is unknown at test time). The class labels are the keys of the
                                dict and the values are numpy arrays of shape `(n_test, n_layers)` similar to
                                `test_stats_pred`.
        :param return_corrected_predictions: Same as the method `score`.
        :return:
        """
        n_test = labels_pred.shape[0]
        # Log of the multivariate p-value estimate of the test statistics under the distribution of each
        # candidate true class
        log_pvalues_true = np.zeros((n_test, self.n_classes))
        for i, c in enumerate(self.labels_unique):
            v = -1. * score_log_normal_mixture(test_stats_true[c], self.density_models_true[c])
            log_pvalues_true[:, i] = np.log(
                pvalue_score(self.samples_neg_log_dens_true[c], v, log_transform=False, bootstrap=False)
            )

        # Adversarial or OOD scores for the test samples and the corrected class predictions
        scores_adver = np.zeros(n_test)
        scores_ood = np.zeros(n_test)
        corrected_classes = copy.copy(labels_pred)
        preds_unique = self.labels_unique if (n_test > 1) else [labels_pred[0]]
        cnt_par = 0
        for c in preds_unique:
            # Scoring samples that are predicted into class `c`
            ind = np.where(labels_pred == c)[0]
            n_pred = ind.shape[0]
            if n_pred == 0:
                continue

            # Score for OOD detection
            v = -1. * score_log_normal_mixture(test_stats_pred[ind, :], self.density_models_pred[c])
            # `pvalue_score` returns negative log of the p-values
            scores_ood[ind] = pvalue_score(self.samples_neg_log_dens_pred[c], v, log_transform=True, bootstrap=False)

            # Mask to include all classes, except the predicted class `c`
            i = np.where(self.labels_unique == c)[0][0]
            mask_excl = np.ones(self.n_classes, dtype=np.bool)
            mask_excl[i] = False
            # Score for adversarial detection
            tmp_arr = log_pvalues_true[ind, :]
            scores_adver[ind] = np.max(tmp_arr[:, mask_excl], axis=1) + scores_ood[ind]

            # Corrected prediction is the class corresponding to the maximum log p-value conditioned that class
            # being the true class
            if return_corrected_predictions:
                corrected_classes[ind] = [self.labels_unique[j] for j in np.argmax(tmp_arr, axis=1)]

            # Break if we have already covered all the test samples
            cnt_par += n_pred
            if cnt_par >= n_test:
                break

        return scores_adver, scores_ood, corrected_classes

    def _score_pvalue_based(self, labels_pred, pvalues_pred, pvalues_true, return_corrected_predictions=False,
                            start_layer=0):
        """
        Scoring method based on combining the p-values of the test statistics calculated from the layer embeddings.

        :param labels_pred: Same as the method `score`.
        :param pvalues_pred: numpy array with the negative log p-values from the different layers and layer pairs,
                             conditioned on the predicted class of the test samples. Is a numpy array of shape
                             `(n_test, n_layers)`, where `n_test` and `n_layers` are the number of test samples
                             and number of layers (layer pairs) respectively.
        :param pvalues_true: dict with the negative log p-values from the different layers, conditioned on each
                             candidate true class (since this is unknown at test time). The class labels are the keys
                             of the dict and the values are numpy arrays of shape `(n_test, n_layers)` similar to
                             `pvalues_pred`.
        :param return_corrected_predictions: Same as the method `score`.
        :param start_layer: Starting index of the layers to include in the p-value fusion. Set to 0 to include all
                            the layers. Set to negative values such as -1, -2, -3 using the same convention as
                            python indexing. For example, a value of `-3` implies the last 3 layers are included.
        :return:
        """
        n_test, nl = pvalues_pred.shape
        # Equal weight to all the layers or layer pairs
        weights = (1. / nl) * np.ones(nl)
        log_weights = np.log(weights)
        mask_layers = np.zeros(nl, dtype=np.bool)
        mask_layers[start_layer:] = True

        # Log of the combined p-values
        pvalues_comb_pred = np.zeros(n_test)
        pvalues_comb_true = np.zeros((n_test, self.n_classes))
        if self.pvalue_fusion == 'fisher':
            pvalues_comb_pred = -1 * np.sum(pvalues_pred[:, mask_layers], axis=1)
            for i, c in enumerate(self.labels_unique):
                pvalues_comb_true[:, i] = -1 * np.sum(pvalues_true[c][:, mask_layers], axis=1)
        elif self.pvalue_fusion == 'harmonic_mean':
            # log of the combined p-values
            arr_temp = log_weights + pvalues_pred
            offset = np.log(np.sum(weights[mask_layers]))
            pvalues_comb_pred = offset - log_sum_exp(arr_temp[:, mask_layers])
            for i, c in enumerate(self.labels_unique):
                arr_temp = log_weights + pvalues_true[c]
                pvalues_comb_true[:, i] = offset - log_sum_exp(arr_temp[:, mask_layers])
        else:
            raise ValueError("Invalid value '{}' for the input argument 'pvalue_fusion'.".format(self.pvalue_fusion))

        # Adversarial or OOD scores for the test samples and the corrected class predictions
        scores_adver = np.zeros(n_test)
        scores_ood = np.zeros(n_test)
        corrected_classes = copy.copy(labels_pred)
        preds_unique = self.labels_unique if (n_test > 1) else [labels_pred[0]]
        cnt_par = 0
        for c in preds_unique:
            # Scoring samples that are predicted into class `c`
            ind = np.where(labels_pred == c)[0]
            n_pred = ind.shape[0]
            if n_pred == 0:
                continue

            # OOD score
            scores_ood[ind] = -1 * pvalues_comb_pred[ind]

            # Adversarial score
            # Mask to include all classes, except the predicted class `c`
            i = np.where(self.labels_unique == c)[0][0]
            mask_excl = np.ones(self.n_classes, dtype=np.bool)
            mask_excl[i] = False
            arr_temp = pvalues_comb_true[ind, :]
            scores_adver[ind] = np.max(arr_temp[:, mask_excl], axis=1) - pvalues_comb_pred[ind]

            # Corrected class prediction based on the maximum p-value conditioned on the candidate true class
            if return_corrected_predictions:
                corrected_classes[ind] = [self.labels_unique[j] for j in np.argmax(arr_temp, axis=1)]

            # Break if we have already covered all the test samples
            cnt_par += n_pred
            if cnt_par >= n_test:
                break

        return scores_adver, scores_ood, corrected_classes

    def _score_klpe(self, labels_pred, test_stats_pred, test_stats_true, return_corrected_predictions=False):
        """
        Scoring method based on the averaged localized p-value estimation method, which estimates the p-value of
        the joint (multivariate) distribution of the test statistics across the layers conditioned on the
        predicted and true class.

        :param labels_pred: Same as the method `score`.
        :param test_stats_pred: numpy array with the test statistics from the different layers, conditioned on the
                                predicted class of the test samples. Is a numpy array of shape `(n_test, n_layers)`,
                                where `n_test` and `n_layers` are the number of test samples and number of layers
                                respectively.
        :param test_stats_true: dict with the test statistics from the different layers, conditioned on each candidate
                                true class (since this is unknown at test time). The class labels are the keys of the
                                dict and the values are numpy arrays of shape `(n_test, n_layers)` similar to
                                `test_stats_pred`.
        :param return_corrected_predictions: Same as the method `score`.
        :return:
        """
        n_test = labels_pred.shape[0]
        # Log of the multivariate p-value estimate of the test statistics under the distribution of each
        # candidate true class
        log_pvalues_true = np.zeros((n_test, self.n_classes))
        for i, c in enumerate(self.labels_unique):
            log_pvalues_true[:, i] = -1. * self.klpe_models_true[c].score(test_stats_true[c])

        # Adversarial or OOD scores for the test samples and the corrected class predictions
        scores_adver = np.zeros(n_test)
        scores_ood = np.zeros(n_test)
        corrected_classes = copy.copy(labels_pred)
        preds_unique = self.labels_unique if (n_test > 1) else [labels_pred[0]]
        cnt_par = 0
        for c in preds_unique:
            # Scoring samples that are predicted into class `c`
            ind = np.where(labels_pred == c)[0]
            n_pred = ind.shape[0]
            if n_pred == 0:
                continue

            # OOD score is the negative log of the multivariate p-value estimate of the test statistics under the
            # distribution of the predicted class `c`
            scores_ood[ind] = self.klpe_models_pred[c].score(test_stats_pred[ind, :])

            # Adversarial score
            # Mask to include all classes, except the predicted class `c`
            i = np.where(self.labels_unique == c)[0][0]
            mask_excl = np.ones(self.n_classes, dtype=np.bool)
            mask_excl[i] = False
            tmp_arr = log_pvalues_true[ind, :]
            scores_adver[ind] = np.max(tmp_arr[:, mask_excl], axis=1) + scores_ood[ind]

            # Corrected prediction is the class corresponding to the maximum log p-value conditioned that class
            # being the true class
            if return_corrected_predictions:
                corrected_classes[ind] = [self.labels_unique[j] for j in np.argmax(tmp_arr, axis=1)]

            # Break if we have already covered all the test samples
            cnt_par += n_pred
            if cnt_par >= n_test:
                break

        return scores_adver, scores_ood, corrected_classes

    def _get_top_ranked(self, test_stats, p_values, reverse=False):
        """
        Get the top-ranked (largest or smallest) test statistics across the layers. Ranking is done based on the
        p-values.

        :param test_stats: numpy array of shape `(n, d)`, where `n` is the number of samples and `d` is the
                           number of test statistics.
        :param p_values: numpy array with the p-values corresponding to the test statistics. Has same shape
                         as `test_stats`.
        :param reverse: set to True to get the largest scores.
        :return:
        """
        if reverse:
            ind = np.fliplr(np.argsort(p_values, axis=1))
        else:
            ind = np.argsort(p_values, axis=1)

        test_stats = np.take_along_axis(test_stats, ind, axis=1)[:, :self.num_top_ranked]
        p_values = np.take_along_axis(p_values, ind, axis=1)[:, :self.num_top_ranked]
        return test_stats, p_values
