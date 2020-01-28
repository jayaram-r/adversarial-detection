
import numpy as np
import torch
from helpers.constants import (
    ROOT,
    SEED_DEFAULT
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

    # `embeddings` will be a list of length equal to the number of layers.
    # `embeddings[i]` will be a list of numpy arrays corresponding to the data batches for layer `i`.
    # `embeddings[i][j]` will have shape `(b, d1, d2, d3)` or `(b, d1)` where `b` is the batch size and the rest
    # are dimensions.
    labels = np.array(labels, dtype=np.int)
    labels_pred = np.array(labels_pred, dtype=np.int)
    # Unique label counts
    labels_uniq, counts = np.unique(labels, return_counts=True)
    for a, b in zip(labels_uniq, counts):
        print("label {}, count = {:d}, proportion = {:.4f}".format(a, b, b / labels.shape[0]))

    if (np.max(counts) / np.min(counts)) >= 1.2:
        print("WARNING: classes are not balanced")

    return embeddings, labels, labels_pred, counts


def transform_layer_embeddings(embeddings_in, transform_models=None, transform_models_file=None):
    """
    Perform dimension reduction on the data embeddings from each layer. The transformation or projection matrix
    for each layer is provided via one of the inputs `transform_models` or `transform_models_file`. Only one of
    them should be specified. In the case of `transform_models_file`, the models are loaded from a pickle file.

    NOTE: In order to not perform dimension reduction at a particular layer, the corresponding element of
    `transform_models` can be set to `None`. Thus, a list of `None` values can be passed to completely skip
    dimension reduction.

    :param embeddings_in: list of data embeddings per layer. `embeddings_in[i]` is a list of numpy arrays
                          corresponding to the data batches from layer `i`.
    :param transform_models: None or a list of dictionaries with the transformation models per layer. The length of
                             `transform_models` should be equal to the length of `embeddings_in`.
    :param transform_models_file: None or a string with the file path. This should be a pickle file with the saved
                                  transformation models per layer.

    :return: list of transformed data arrays, one per layer.
    """
    if transform_models_file is None:
        if transform_models is None:
            raise ValueError("Both inputs 'transform_models' and 'transform_models_file' are not specified.")

    else:
        transform_models = load_dimension_reduction_models(transform_models_file)

    n_layers = len(embeddings_in)
    assert len(transform_models) == n_layers, ("Length of 'transform_models' is not equal to the length of "
                                               "'embeddings_in'")
    embeddings_out = []
    for i in range(n_layers):
        print("Transforming embeddings from layer {:d}".format(i + 1))
        data_in = combine_and_vectorize(embeddings_in[i])
        embeddings_out.append(transform_data_from_model(data_in, transform_models[i]))

    return embeddings_out
