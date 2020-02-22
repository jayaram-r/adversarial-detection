
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
    TEST_STATS_SUPPORTED
)
from helpers.dimension_reduction_methods import (
    transform_data_from_model,
    load_dimension_reduction_models
)
from helpers.test_statistics_layers import (
    MultinomialScore,
    LIDScore,
    LLEScore
)
from helpers.density_model_layer_statistics import (
    train_log_normal_mixture,
    score_log_normal_mixture
)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def combine_and_vectorize(data_batches):
    """
    Combine a list of data batches and vectorize them if they are tensors. If there is only a single data batch,
    it can be passed in as list with a single array.

    :param data_batches: list of numpy arrays containing the data batches. Each array has shape `(n, d1, ...)`,
                         where `n` can be different across the batches, but the remaining dimensions should be
                         the same.
    :return: single numpy array with the combined, vectorized data.
    """
    data = np.concatenate(data_batches, axis=0)
    s = data.shape
    if len(s) > 2:
        data = data.reshape((s[0], -1))

    return data


def extract_layer_embeddings(model, device, data_loader, method='proposed', num_samples=None):
    """
    Extract the layer embeddings produced by a trained DNN model on the given data set. Also, returns the true class
    and the predicted class for each sample.

    :param model: torch NN model.
    :param device: torch device type - cuda or cpu.
    :param data_loader: torch data loader object which is an instancee of `torch.utils.data.DataLoader`.
    :param method: string with the name of the proposed method. Valid choices are ['proposed', 'odds', 'lid'].
    :param num_samples: None or an int value specifying the number of samples to select.

    :return:
        - embeddings: list of numpy arrays, one per layer, where the i-th array has shape `(N, d_i)`, `N` being
                      the number of samples and `d_i` being the vectorized dimension of layer `i`.
        - labels: numpy array of class labels. Has shape `(N, )`.
        - labels_pred: numpy array of the model-predicted class labels. Has shape `(N, )`.
        - counts: numpy array of sample counts for each distinct class in `labels`.
    """
    if model.training:
        model.eval()

    labels = []
    labels_pred = []
    embeddings = []
    num_samples_partial = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            # target = target.to(device)

            temp = target.detach().cpu().numpy()
            labels.extend(temp)
            num_samples_partial += temp.shape[0]
            # print(batch_idx)

            # Predicted class
            outputs = model(data)
            _, predicted = outputs.max(1)
            labels_pred.extend(predicted.detach().cpu().numpy())
            # Layer outputs
            if method in ('proposed', 'dknn', 'trust'):
                outputs_layers = model.layer_wise(data)
            elif method == 'odds':
                outputs_layers = model.layer_wise_odds_are_odd(data)
            elif method == 'lid':
                outputs_layers = model.layer_wise_lid_method(data)
            else:
                raise ValueError("Invalid value '{}' for input 'method'".format(method))

            if batch_idx > 0:
                for i in range(len(outputs_layers)):    # each layer
                    embeddings[i].append(outputs_layers[i].detach().cpu().numpy())
            else:
                embeddings = [[v.detach().cpu().numpy()] for v in outputs_layers]

            if num_samples:
                if num_samples_partial >= num_samples:
                    break

    '''
    `embeddings` will be a list of length equal to the number of layers.
    `embeddings[i]` will be a list of numpy arrays corresponding to the data batches for layer `i`.
    `embeddings[i][j]` will be an array of shape `(b, d1, d2, d3)` or `(b, d1)` where `b` is the batch size
     and the rest are dimensions.
    '''
    embeddings = [combine_and_vectorize(v) for v in embeddings]

    labels = np.array(labels, dtype=np.int)
    labels_pred = np.array(labels_pred, dtype=np.int)
    # Unique label counts
    print("\nNumber of labeled samples per class:")
    labels_uniq, counts = np.unique(labels, return_counts=True)
    for a, b in zip(labels_uniq, counts):
        print("class {}, count = {:d}, proportion = {:.4f}".format(a, b, b / labels.shape[0]))

    # if (np.max(counts) / np.min(counts)) >= 1.2:
    #    logger.warning("Classes are not balanced.")

    print("\nNumber of predicted samples per class:")
    preds_uniq, counts_pred = np.unique(labels_pred, return_counts=True)
    for a, b in zip(preds_uniq, counts_pred):
        print("class {}, count = {:d}, proportion = {:.4f}".format(a, b, b / labels_pred.shape[0]))

    if preds_uniq.shape[0] != labels_uniq.shape[0]:
        logger.error("Number of unique predicted classes is not the same as the number of labeled classes.")

    return embeddings, labels, labels_pred, counts


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
    """
    def __init__(self,
                 layer_statistic='multinomial',
                 use_top_ranked=False,
                 num_top_ranked=NUM_TOP_RANKED,
                 skip_dim_reduction=False,
                 model_file_dim_reduction=None,
                 neighborhood_constant=NEIGHBORHOOD_CONST, n_neighbors=None,
                 metric=METRIC_DEF, metric_kwargs=None,
                 approx_nearest_neighbors=True,
                 n_jobs=1,
                 low_memory=False,
                 seed_rng=SEED_DEFAULT):
        """

        :param layer_statistic: Type of test statistic to calculate at the layers. Valid values are 'multinomial',
                                'lid', and 'lle'.
        :param use_top_ranked: Set to True in order to use only a few top ranked test statistics for detection.
        :param num_top_ranked: If `use_top_ranked` is set to True, this specifies the number of top-ranked test
                               statistics to use for detection. This number should be smaller than the number of
                               layers considered for detection.
        :param skip_dim_reduction: Set to True in order to skip dimension reduction of the layer embeddings.
        :param model_file_dim_reduction: Path to the model file that contains the models for performing dimension
                                         reduction at each layerr. This will be a pickle file that loads into a list
                                         of model dictionaries.
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
        self.use_top_ranked = use_top_ranked
        self.num_top_ranked = num_top_ranked
        self.skip_dim_reduction = skip_dim_reduction
        self.model_file_dim_reduction = model_file_dim_reduction
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

        if self.layer_statistic in {'lid', 'lle'}:
            if not self.skip_dim_reduction:
                logger.warning("Option 'skip_dim_reduction' is set to False for the test statistic '{}'. Setting "
                               "it to True because it is preferred to skip dimension reduction for this "
                               "test statistic.".format(self.layer_statistic))
                self.skip_dim_reduction = True

        # Load the dimension reduction models per-layer if required
        self.transform_models = None
        if not self.skip_dim_reduction:
            if self.model_file_dim_reduction is None:
                raise ValueError("Model file for dimension reduction is required but not specified as input.")

            self.transform_models = load_dimension_reduction_models(self.model_file_dim_reduction)

        self.n_layers = None
        self.labels_unique = None
        self.n_classes = None
        self.n_samples = None
        # List of test statistic model instances for each layer
        self.test_stats_models = []
        # dict mapping each class `c` to the joint density model of the test statistics conditioned on predicted
        # class being `c`
        self.density_models_pred = dict()
        # dict mapping each class `c` to the joint density model of the test statistics conditioned on true
        # class being `c`
        self.density_models_true = dict()
        # Log of the class prior probabilities estimated from the training data labels
        self.log_class_priors = None

    def fit(self, layer_embeddings, labels, labels_pred):
        """
        Estimate parameters of the detection method given natural (non-adversarial) input data.
        NOTE: Inputs to this method can be obtained by calling the function `extract_layer_embeddings`.

        :param layer_embeddings: list of numpy arrays with the layer embedding data. Length of the list is equal to
                                 the number of layers. The numpy array at index `i` has shape `(n, d_i)`, where `n`
                                 is the number of samples and `d_i` is the dimension of the embeddings at layer `i`.
        :param labels: numpy array of labels for the classification problem addressed by the DNN. Should have shape
                       `(n, )`, where `n` is the number of samples.
        :param labels_pred: numpy array of class predictions made by the DNN. Should have the same shape as `labels`.

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
        test_stats_pred = dict()
        for c in self.labels_unique:
            indices_true[c] = np.where(labels == c)[0]
            indices_pred[c] = np.where(labels_pred == c)[0]
            # Test statistics across the layers for the samples labeled into class `c`
            test_stats_true[c] = np.zeros((indices_true[c].shape[0], self.n_layers))
            # Test statistics across the layers for the samples predicted into class `c`
            test_stats_pred[c] = np.zeros((indices_pred[c].shape[0], self.n_layers))
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

            logger.info("Parameter estimatison and test statistics calculation for layer {:d}:".format(i + 1))
            ts_obj = None
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
            elif self.layer_statistic == 'lid':
                ts_obj = LIDScore(
                    neighborhood_constant=self.neighborhood_constant,
                    n_neighbors=self.n_neighbors,
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

            scores_temp = ts_obj.fit(data_proj, labels, labels_pred, labels_unique=self.labels_unique)
            '''
            `scores_temp` will be a numpy array of shape `(self.n_samples, self.n_classes + 1)` with a vector of 
            test statistics for each sample.
            The first column `scores_temp[:, 0]` gives the scores conditioned on the predicted class.
            The remaining columns `scores_temp[:, i]` for `i = 1, 2, . . .` gives the scores conditioned on `i - 1`
            being the candidate true class for the sample.
            '''
            self.test_stats_models.append(ts_obj)
            for j, c in enumerate(self.labels_unique):
                # Test statistics from layer `i`
                test_stats_pred[c][:, i] = scores_temp[indices_pred[c], 0]
                test_stats_true[c][:, i] = scores_temp[indices_true[c], j + 1]

        # Learn a joint probability density model for the test statistics
        for c in self.labels_unique:
            # Get the top ranked order statistics if required
            if self.use_top_ranked:
                logger.info("Using the largest (smallest) {:d} test statistics conditioned on the predicted "
                            "(true) class.".format(self.num_top_ranked))
                # For the test statistics conditioned on the predicted class, take the largest `self.num_top_ranked`
                # test statistics across the layers
                test_stats_pred[c] = self.get_top_ranked(test_stats_pred[c], reverse=True)

                # For the test statistics conditioned on the true class, take the smallest `self.num_top_ranked`
                # test statistics across the layers
                test_stats_true[c] = self.get_top_ranked(test_stats_true[c])

            logger.info("Learning a joint probability density model for the test statistics conditioned on the "
                        "predicted class '{}':".format(c))
            logger.info("Number of samples = {:d}, dimension = {:d}".format(*test_stats_pred[c].shape))
            self.density_models_pred[c] = train_log_normal_mixture(test_stats_pred[c], seed_rng=self.seed_rng)

            logger.info("Learning a joint probability density model for the test statistics conditioned on the "
                        "true class '{}':".format(c))
            logger.info("Number of samples = {:d}, dimension = {:d}".format(*test_stats_true[c].shape))
            self.density_models_true[c] = train_log_normal_mixture(test_stats_true[c], seed_rng=self.seed_rng)

        return self

    def score(self, layer_embeddings, labels_pred, return_corrected_predictions=False, is_train=False):
        """
        Given the layer embeddings (including possibly the input itself) and the predicted classes for test data,
        score them on how likely they are to be adversarial or out-of-distribution (OOD). Larger values of the
        scores correspond to a higher probability that the test sample is adversarial or OOD. The scores can be
        thresholded, with values above the threshold declared as adversarial or OOD. The threshold can be set such
        that the detector has a target false positive rate.

        :param layer_embeddings: list of numpy arrays with the layer embedding data. Length of the list is equal to
                                 the number of layers. The numpy array at index `i` has shape `(n, d_i)`, where `n`
                                 is the number of samples and `d_i` is the dimension of the embeddings at layer `i`.
        :param labels_pred: numpy array of class predictions made by the DNN. Should have the same shape as `labels`.
        :param return_corrected_predictions: Set to True in order to get the most probable class prediction based
                                             on Bayes class posterior given the test statistic vector. Note that this
                                             will change the returned values.
        :param is_train: Set to True if the inputs are the same non-adversarial inputs used with the `fit` method.

        :return: (scores_adver, scores_ood [, corrected_classes])
            - scores_adver: numpy array of scores corresponding to attack detection. The array should have shape
                            `(labels_pred.shape[0], )`. Larger values of the scores correspond to a higher probability
                            that the test sample is adversarial.
            - scores_ood: numpy array of scores corresponding to OOD detection. Same shape as `scores_adver`. Larger
                          values of the scores correspond to a higher probability that the test sample is OOD.
            #
            # returned only if `return_corrected_predictions = True`
            - corrected_classes: numpy array of the corrected class predictions. Has same shape and dtype as the
                                 array `labels_pred`.
        """
        n_test = labels_pred.shape[0]
        l = len(layer_embeddings)
        if l != self.n_layers:
            raise ValueError("Expecting {:d} layers in the input data, but received {:d}".format(self.n_layers, l))

        # Test statistics at each layer conditioned on the predicted class and candidate true classes
        test_stats_pred = np.zeros((n_test, self.n_layers))
        test_stats_true = {c: np.zeros((n_test, self.n_layers)) for c in self.labels_unique}
        for i in range(self.n_layers):
            if self.transform_models:
                # Dimension reduction
                data_proj = transform_data_from_model(layer_embeddings[i], self.transform_models[i])
            else:
                data_proj = layer_embeddings[i]

            # Test statistics for layer `i`
            scores_temp = self.test_stats_models[i].score(data_proj, labels_pred, is_train=is_train)
            # `scores_test` will have shape `(n_test, self.n_classes + 1)`

            test_stats_pred[:, i] = scores_temp[:, 0]
            for j, c in enumerate(self.labels_unique):
                test_stats_true[c][:, i] = scores_temp[:, j + 1]

        if self.use_top_ranked:
            # For the test statistics conditioned on the predicted class, take the largest `self.num_top_ranked`
            # test statistics across the layers
            test_stats_pred = self.get_top_ranked(test_stats_pred, reverse=True)

            for c in self.labels_unique:
                # For the test statistics conditioned on the true class, take the smallest `self.num_top_ranked`
                # test statistics across the layers
                test_stats_true[c] = self.get_top_ranked(test_stats_true[c])

        # Adversarial or OOD scores for the test samples
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

            # OOD score:
            # $- log p(t_1, \cdots, t_L | \hat{C} = c)$
            s1 = score_log_normal_mixture(test_stats_pred[ind, :], self.density_models_pred[c])
            scores_ood[ind] = -s1

            # Adversarial score:
            # $max_{k \neq c} log p(t_1, \cdots, t_L | C = k) - log p(t_1, \cdots, t_L | \hat{C} = c)$
            s2 = np.zeros((n_pred, self.n_classes - 1))
            if return_corrected_predictions:
                logit_scores = np.zeros((n_pred, self.n_classes))
            else:
                logit_scores = None

            jj = 0
            for j, k in enumerate(self.labels_unique):
                if k != c:
                    s2[:, jj] = score_log_normal_mixture(test_stats_true[k][ind, :], self.density_models_true[k])
                    if return_corrected_predictions:
                        logit_scores[:, j] = self.log_class_priors[k] + s2[:, jj]

                    jj += 1
                else:
                    if return_corrected_predictions:
                        logit_scores[:, j] = self.log_class_priors[k] + \
                                             score_log_normal_mixture(test_stats_true[k][ind, :],
                                                                      self.density_models_true[k])

            # Score for adversarial detection
            scores_adver[ind] = np.max(s2, axis=1) - s1
            # Corrected class prediction based on the logit scores (for samples predicted into class c)
            if return_corrected_predictions:
                corrected_classes[ind] = [self.labels_unique[j] for j in np.argmax(logit_scores, axis=1)]

            # Break if we have already covered all the test samples
            cnt_par += n_pred
            if cnt_par >= n_test:
                break

        if return_corrected_predictions:
            return scores_adver, scores_ood, corrected_classes
        else:
            return scores_adver, scores_ood

    def get_top_ranked(self, test_stats, reverse=False):
        """
        Get the top-ranked (largest or smallest) test statistics across the layers.

        :param test_stats: numpy array of shape `(n, d)`, where `n` is the number of samples and `d` is the
                           number of test statistics.
        :param reverse: set to True to get the largest scores.
        :return:
        """
        if reverse:
            return np.fliplr(np.sort(test_stats, axis=1))[:, :self.num_top_ranked]
        else:
            return np.sort(test_stats, axis=1)[:, :self.num_top_ranked]
