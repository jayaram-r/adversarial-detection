
import numpy as np
import torch
import logging
from helpers.constants import (
    ROOT,
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
    LIDScore
)
from helpers.density_model_layer_statistics import (
    train_log_normal_mixture,
    score_log_normal_mixture
)

logging.basicConfig(level=logging.INFO)
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


def extract_layer_embeddings(model, device, data_loader, num_samples=None):
    """
    Extract the layer embeddings produced by a trained DNN model on the given data set. Also, returns the true class
    and the predicted class for each sample.

    :param model: torch NN model.
    :param device: torch device type - cuda or cpu.
    :param data_loader: torch data loader object which is an instancee of `torch.utils.data.DataLoader`.
    :param num_samples: None or an int value specifying the number of samples to select.

    :return:
        - embeddings: list of numpy arrays, one per layer, where the i-th array has shape `(N, d_i)`, `N` being
                      the number of samples and `d_i` being the vectorized dimension of layer `i`.
        - labels: numpy array of class labels. Has shape `(N, )`.
        - labels_pred: numpy array of the model-predicted class labels. Has shape `(N, )`.
        - counts: numpy array of sample counts for each distinct class in `labels`.
    """
    labels = []
    labels_pred = []
    embeddings = []
    num_samples_partial = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            temp = target.detach().cpu().numpy()
            labels.extend(temp)
            num_samples_partial += temp.shape[0]
            # print(batch_idx)

            # Predicted class
            outputs = model(data)
            _, predicted = outputs.max(1)
            labels_pred.extend(predicted.detach().cpu().numpy())
            # Layer outputs
            outputs_layers = model.layer_wise(data)
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

        if self.layer_statistic not in TEST_STATS_SUPPORTED:
            raise ValueError("Invalid value '{}' for the input argument 'layer_statistic'.".
                             format(self.layer_statistic))

        if self.layer_statistic == 'lid':
            if not self.skip_dim_reduction:
                logger.warning("Option 'skip_dim_reduction' is set to False for the LID test statistic. Setting it to "
                               "True because it is preferred to skip dimension reduction for this test statistic.")
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

    def fit(self, layer_embeddings, labels, labels_pred):
        self.n_layers = len(layer_embeddings)
        self.labels_unique = np.unique(labels)
        self.n_classes = len(self.labels_unique)
        self.n_samples = labels.shape[0]

        logger.info("Number of classes: {:d}.".format(self.n_classes))
        logger.info("Number of layer embeddings: {:d}".format(self.n_layers))
        logger.info("Number of samples: {:d}.".format(self.n_samples))
        if labels_pred.shape[0] != self.n_samples:
            raise ValueError("Inputs 'labels' and 'labels_pred' do not have the same size.")

        if not all([layer_embeddings[i].shape[0] == self.n_samples for i in range(self.n_layers)]):
            raise ValueError("Input 'layer_embeddings' does not have the expected format")

        if self.use_top_ranked:
            if self.num_top_ranked > self.n_layers:
                logger.warning("Number of top-ranked layer statistics cannot be larger than the number of layers. "
                               "Setting it equal to the number of layers ({:d}).".format(self.n_layers))
                self.num_top_ranked = self.n_layers

        if self.transform_models:
            # Dimension reduction at each via a layer-specific linear transformation
            layer_embeddings = transform_layer_embeddings(layer_embeddings, self.transform_models)

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

        for i in range(self.n_layers):
            logger.info("Parameter estimation and test statistics calculation for layer {:d}:".format(i + 1))
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

            scores_temp = ts_obj.fit(layer_embeddings[i], labels, labels_pred,
                                     labels_unique=self.labels_unique)
            '''
            `scores_temp` will be a numpy array of shape `(self.n_samples, self.n_classes + 1)` with a vector of 
            test statistics for each sample.
            The first column `scores_temp[:, 0]` gives the scores conditioned on the predicted class.
            The remaining columns `scores_temp[:, i]` for `i = 1, 2, . . .` gives the scores conditioned on `i - 1`
            being the candidate true class for the sample.
            '''
            for j, c in enumerate(self.labels_unique):
                # Test statistics from layer `i`
                test_stats_pred[c][:, i] = scores_temp[indices_pred[c], 0]
                test_stats_true[c][:, i] = scores_temp[indices_true[c], j + 1]

        # Get the top ranked order statistics if required
        if self.use_top_ranked:
            for c in self.labels_unique:
                # For the test statistics conditioned on the predicted class, take the largest `self.num_top_ranked`
                # test statistics across the layers
                arr = np.fliplr(np.sort(test_stats_pred[c], axis=1))
                test_stats_pred[c] = arr[:, :self.num_top_ranked]

                # For the test statistics conditioned on the true class, take the smallest `self.num_top_ranked`
                # test statistics across the layers
                arr = np.sort(test_stats_true[c], axis=1)
                test_stats_true[c] = arr[:, :self.num_top_ranked]

        # Learn a probability density model for the test statistics
