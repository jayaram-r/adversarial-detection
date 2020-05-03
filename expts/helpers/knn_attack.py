# Implementation of the custom KNN attack defined in the paper
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import os
import sys
import multiprocessing
from functools import partial
from sklearn.metrics import pairwise_distances
from scipy.special import softmax, logsumexp
from helpers.utils import (
    combine_and_vectorize,
    get_data_bounds
)
from helpers.knn_index import KNNIndex

INFTY = 1e20
NORM_REG = 1e-16


def get_labels(data_loader):
    # function to return an ndarray with all the labels of the data_loader, and will return a list with the smallest
    # and largest label
    labels = []
    for data, target in data_loader:
        labels.extend(target.detach().cpu().numpy())

    labels = np.array(labels, dtype=np.int)
    # Find the unique labels in sorted order
    labels_uniq = np.unique(labels)

    return labels, labels_uniq


def extract_layer_embeddings(model, device, data_loader, indices, split_by_class=True):
    # returns a list of torch tensors, where each torch tensor is a per layer embedding
    if model.training:
        model.eval()

    embeddings = []
    n_layers = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            # Layer outputs
            outputs_layers = model.layer_wise(data)

            if batch_idx > 0:
                for i in range(n_layers):
                    embeddings[i].append(outputs_layers[i].detach().cpu().numpy())
            else:
                # First batch
                n_layers = len(outputs_layers)
                embeddings = [[v.detach().cpu().numpy()] for v in outputs_layers]

    if split_by_class:
        # Dict mapping each class label to a list of torch tensors per layer, where each tensor includes only
        # samples from that class
        output = {}
        for i in range(n_layers):
            # Combine all the data batches and convert the numpy array to torch tensor
            embeddings_comb = combine_and_vectorize(embeddings[i])

            for label in indices.keys():
                temp_arr = torch.from_numpy(embeddings_comb[indices[label], :]).to(device)
                if i == 0:
                    output[label] = [temp_arr]
                else:
                    output[label].append(temp_arr)

    else:
        # List of torch tensors per layer
        output = []
        for i in range(n_layers):
            # Combine all the data batches and convert the numpy array to torch tensor
            temp_arr = torch.from_numpy(combine_and_vectorize(embeddings[i])).to(device)
            output.append(temp_arr)

    return output


def extract_input_embeddings(model, device, x_input):
    # Returns a list of torch tensors, where each torch tensor is a per layer embedding
    # Model should already be set to eval mode
    # if model.training:
    #    model.eval()

    data = x_input.to(device)
    # Layer outputs
    outputs_layers = model.layer_wise(data)

    n_layers = len(outputs_layers)
    for i in range(n_layers):
        # Flatten all but the first dimension of the tensor and set/unset its requires_grad flag
        tens = outputs_layers[i]
        s = tens.size()
        if len(s) > 2:
            tens = tens.view(s[0], -1)

        outputs_layers[i] = tens

    return outputs_layers


def get_predicted_class(model, device, x_input):
    # Get the DNN model predictions on a batch of inputs
    with torch.no_grad():
        x = x_input.to(device)
        _, preds = model(x).max(1)

    return preds.cpu().numpy()


def get_runner_up_class(model, device, x_input):
    # Get the second best DNN model predictions on a batch of inputs
    with torch.no_grad():
        x = x_input.to(device)
        y = model(x)
        ret = torch.argsort(y, dim=1)

    return ret[:, -2].cpu().numpy()


def objective_kernel_scale(dist_neighbors, dist_all, alpha, sigma):
    """
    Objective function that is to be maximized for selecting the kernel scale.

    :param dist_neighbors: numpy array with the pairwise distances between test samples and their `k` nearest
                           neighbors. Has shape like `(n_test, k)`.
    :param dist_all: numpy array with the pairwise distances between test samples and all the representative samples.
                     Has shape like `(n_test, n_reps)`.
    :param alpha: float value between 0 and 1 specifying the relative weight of the two terms in the objective
                  function.
    :param sigma: numpy array of kernel sigma values. Has shape like `(n_test, )`.

    :return: objective function value for each of the test samples. Has shape `(n_test, )`.
    """
    sigma = sigma[:, np.newaxis]
    expo_arr_neigh = np.negative((dist_neighbors / sigma) ** 2)
    temp_arr1 = softmax(expo_arr_neigh, axis=1)
    # Entropy normalized to the range [0, 1]
    entropy = (np.sum(temp_arr1 * np.log(np.clip(temp_arr1, sys.float_info.min, None)), axis=1) /
               np.log(1. / dist_neighbors.shape[1]))
    expo_arr_all = np.negative((dist_all / sigma) ** 2)
    proba = np.exp(logsumexp(expo_arr_neigh, axis=1) - logsumexp(expo_arr_all, axis=1))

    return (1. - alpha) * proba + alpha * entropy


# Helper function used by multiprocessing
def helper_objective(dist_neighbors, dist_all, alpha, sigma_cand_vals, index_cand):
    return objective_kernel_scale(dist_neighbors, dist_all, alpha, sigma_cand_vals[:, index_cand])


def set_kernel_scale(layer_embeddings_train, layer_embeddings_test, metric='euclidean', n_neighbors=10,
                     n_jobs=1, search_size=20, alpha=0.5):
    # `layer_embeddings_train` and `layer_embeddings_test` will both be a list of numpy arrays
    n_layers = len(layer_embeddings_test)
    n_test = layer_embeddings_test[0].shape[0]
    # n_train = layer_embeddings_train[0].shape[0]

    # `1 - epsilon` values
    v = np.linspace(0.05, 0.95, num=search_size)
    sigma_multiplier = np.sqrt(-1. / np.log(v))
    sigma_per_layer = np.ones((n_test, n_layers))
    for i in range(n_layers):
        if metric == 'cosine':
            # For cosine distance, we scale the layer embedding vectors to have unit norm
            norm_train = np.linalg.norm(layer_embeddings_train[i], axis=1) + NORM_REG
            x_train = layer_embeddings_train[i] / norm_train[:, np.newaxis]
            norm_test = np.linalg.norm(layer_embeddings_test[i], axis=1) + NORM_REG
            x_test = layer_embeddings_test[i] / norm_test[:, np.newaxis]
        else:
            x_train = layer_embeddings_train[i]
            x_test = layer_embeddings_test[i]

        # Build a KNN index on the layer embeddings from the train split
        index_knn = KNNIndex(
            x_train,
            n_neighbors=n_neighbors,
            metric='euclidean',
            approx_nearest_neighbors=True,
            n_jobs=n_jobs
        )
        # Query the index of nearest neighbors of the layer embeddings from the test split
        nn_indices, nn_distances = index_knn.query(x_test, k=n_neighbors)
        # `nn_indices` and `nn_distances` should have shape `(n_test, n_neighbors)`

        # Candidate sigma values are obtained by multiplying `sqrt(\eta_k^2 - \eta_1^2)` of each test point with
        # the `sigma_multiplier` defined earlier. Here `eta_k` and `eta_1` denote distance to the k-th and the 1-st
        # nearest neighbor respectively
        sigma_cand_vals = (np.sqrt(nn_distances[:, -1] ** 2 - nn_distances[:, 0] ** 2).reshape(n_test, 1) *
                           sigma_multiplier.reshape(1, search_size))
        # `sigma_cand_vals` should have shape `(n_test, search_size)`

        # Compute pairwise distances between points in `layer_embeddings_test` and `layer_embeddings_train`
        dist_mat = pairwise_distances(
            x_test,
            Y=x_train,
            metric='euclidean',
            n_jobs=n_jobs
        )
        # `dist_mat` should have shape `(n_test, n_train)`
        # Calculate the objective function to maximize for different candidate `sigma` values
        if n_jobs == 1:
            out = [helper_objective(nn_distances, dist_mat, alpha, sigma_cand_vals, t) for t in range(search_size)]
        else:
            # partial function called by multiprocessing
            helper_objective_partial = partial(helper_objective, nn_distances, dist_mat, alpha, sigma_cand_vals)
            pool_obj = multiprocessing.Pool(processes=n_jobs)
            out = []
            _ = pool_obj.map_async(helper_objective_partial, range(search_size), callback=out.extend)
            pool_obj.close()
            pool_obj.join()

        # `out` will be a list of length `search_size`, where each element is a numpy array with the objective
        # function values for the `n_test` samples.
        # `objec_arr` will have shape `(search_size, n_test)`
        objec_arr = np.array(out)
        # Find the sigma value corresponding to the maximum objective function for each test sample
        ind_max = np.argmax(objec_arr, axis=0)
        sigma_per_layer[:, i] = [sigma_cand_vals[j, ind_max[j]] for j in range(n_test)]

    return sigma_per_layer


def log_sum_gaussian_kernels(x, reps, sigma, metric):
    """
    Compute the log of the sum of Gaussian kernels evaluated at the set of representative points `reps` and
    centered on `x`.

    :param x: torch tensor with a particular layer embedding of the perturbed inputs. Specifically, `f_l(x + delta)`.
              It is expected to be a tensor of shape `(n_samp, n_dim)`, where `n_samp` is the number of inputs and
              `n_dim` is the number of dimensions.
    :param reps: torch tensor with the layer embeddings of the representative samples from a particular class.
                 Specifically, `f_l(x_n)` for the set of `n` from a given class. Expected to be a tensor of shape
                 `(n_reps, n_dim)`, where `n_reps` is the number of representative samples and `n_dim` is the number
                 of dimensions.
    :param sigma: torch tensor with the scale or standard deviation values of the Gaussian kernel. Should have
                  shape `(n_samp, )`.
    :param metric: distance metric. Set to 'euclidean' or 'cosine'.

    :return: torch tensor of shape `(n_samp, )` with the log sum of the Gaussian kernels for each of the `n_samp`
             inputs in `x`.
    """
    n_samp, d = x.size()
    n_reps, n_dim = reps.size()
    assert d == n_dim, "Mismatch in the input dimensions"

    if metric == 'cosine':
        # Rescale the vectors to unit norm, which makes the squared euclidean distance equal to twice the
        # cosine distance
        norm_x = torch.norm(x, p=2, dim=1) + NORM_REG
        x_n = torch.div(x, norm_x.view(n_samp, 1))
        norm_reps = torch.norm(reps, p=2, dim=1) + NORM_REG
        reps_n = torch.div(reps, norm_reps.view(n_reps, 1))
    elif metric == 'euclidean':
        x_n = x
        reps_n = reps
    else:
        raise ValueError("Invalid value '{}' for the input 'metric'".format(metric))

    # Compute pairwise euclidean distances
    dist_mat = torch.squeeze(torch.cdist(torch.unsqueeze(x_n, 0), torch.unsqueeze(reps_n, 0), p=2), 0)
    temp_ten = -1. * torch.pow(torch.div(dist_mat, sigma.view(n_samp, 1)), 2)
    # `dist_mat` and `temp_ten` will have the same size `(n_samp, n_reps)`
    with torch.no_grad():
        max_val, _ = torch.max(temp_ten, dim=1)
        # `max_val` will have size `(n_samp, )`

    # numerically stable computation of log-sum-exp
    return torch.log(torch.exp(temp_ten - torch.unsqueeze(max_val, 1)).sum(1)) + max_val


def loss_function_targeted(x, x_recon, x_embeddings, reps, input_indices, target_indices, n_layers, device,
                           const, sigma, dist_metric='euclidean'):
    # Loss function the targeted attack
    batch_size = x.size(0)
    # first double summation based on the paper for the original class `c`
    adv_loss1 = torch.zeros(batch_size, device=device)
    for c, ind_c in input_indices.items():
        temp_sum1 = torch.zeros(ind_c.shape[0], n_layers, device=device)
        for i in range(n_layers):
            # log-sum of kernels from layer `i` for the samples from class `c`
            temp_sum1[:, i] = log_sum_gaussian_kernels(x_embeddings[i][ind_c, :], reps[c][i], sigma[ind_c, i],
                                                       dist_metric)

        with torch.no_grad():
            max_val1, _ = torch.max(temp_sum1, dim=1)

        adv_loss1[ind_c] = torch.log(torch.exp(temp_sum1 - torch.unsqueeze(max_val1, 1)).sum(1)) + max_val1 - \
                           math.log(reps[c][0].size(0))

    # second double summation based on the paper for the target class `c_prime`
    adv_loss2 = torch.zeros(batch_size, device=device)
    for c_prime, ind_c in target_indices.items():
        temp_sum2 = torch.zeros(ind_c.shape[0], n_layers, device=device)
        for i in range(n_layers):
            # log-sum of kernels from layer `i` for the samples from class `c_prime`
            temp_sum2[:, i] = log_sum_gaussian_kernels(x_embeddings[i][ind_c, :], reps[c_prime][i], sigma[ind_c, i],
                                                       dist_metric)

        with torch.no_grad():
            max_val2, _ = torch.max(temp_sum2, dim=1)

        adv_loss2[ind_c] = torch.log(torch.exp(temp_sum2 - torch.unsqueeze(max_val2, 1)).sum(1)) + max_val2 - \
                           math.log(reps[c_prime][0].size(0))

    # distance between the original and perturbed inputs
    dist = ((x - x_recon).view(batch_size, -1) ** 2).sum(1)

    # `const` multiplies the loss term to be consistent with CW attack formulation.
    # Smaller values of `const` lead to solutions with smaller perturbation norm
    total_loss = dist.to(device) + const * (adv_loss1 - adv_loss2)

    return total_loss.mean(), dist.sqrt()


def loss_function_untargeted(x, x_recon, x_embeddings, reps, input_indices, labels_uniq, n_reps, n_layers, device,
                             const, sigma, dist_metric='euclidean'):
    # Loss function the untargeted attack
    batch_size = x.size(0)
    n_classes = labels_uniq.shape[0]

    # double summations based on the paper for the original class `c`
    adv_loss1 = torch.zeros(batch_size, device=device)
    adv_loss2 = torch.zeros(batch_size, device=device)
    for c, ind_c in input_indices.items():
        temp_sum1 = torch.zeros(ind_c.shape[0], n_layers, device=device)
        temp_sum2 = torch.zeros(ind_c.shape[0], n_layers * (n_classes - 1), device=device)
        j = 0
        for i in range(n_layers):
            # log-sum of kernels from layer `i` for the samples from class `c`
            temp_sum1[:, i] = log_sum_gaussian_kernels(x_embeddings[i][ind_c, :], reps[c][i], sigma[ind_c, i],
                                                       dist_metric)
            # log-sum of kernels from layer `i` for the samples from all classes excluding `c`
            for c_prime in labels_uniq:
                if c_prime != c:
                    temp_sum2[:, j] = log_sum_gaussian_kernels(x_embeddings[i][ind_c, :], reps[c_prime][i],
                                                               sigma[ind_c, i], dist_metric)
                    j += 1

        with torch.no_grad():
            max_val1, _ = torch.max(temp_sum1, dim=1)
            max_val2, _ = torch.max(temp_sum2, dim=1)

        adv_loss1[ind_c] = torch.log(torch.exp(temp_sum1 - torch.unsqueeze(max_val1, 1)).sum(1)) + max_val1 - \
                           math.log(reps[c][0].size(0))
        adv_loss2[ind_c] = torch.log(torch.exp(temp_sum2 - torch.unsqueeze(max_val2, 1)).sum(1)) + max_val2 - \
                           math.log(n_reps - reps[c][0].size(0))

    # distance between the original and perturbed inputs
    dist = ((x - x_recon).view(batch_size, -1) ** 2).sum(1)

    # `const` multiplies the loss term to be consistent with CW attack formulation.
    # Smaller values of `const` lead to solutions with smaller perturbation norm
    total_loss = dist.to(device) + const * (adv_loss1 - adv_loss2)

    return total_loss.mean(), dist.sqrt()
    

def check_adv_detec(x_embeddings, labels, labels_pred_dnn, model_detector, is_numpy=False):
    if not is_numpy:
        # If `x_embeddings` is a list of torch tensors, it needs to be converted into a list of numpy arrays
        # before calling the detector's score method
        x_embeddings = [tens.detach().cpu().numpy() for tens in x_embeddings]

    # scores and class predictions from the detection model
    if model_detector._name == 'dknn':
        _, labels_pred = model_detector.score(x_embeddings)
    elif model_detector._name == 'proposed':
        _, labels_pred = model_detector.score(x_embeddings, labels_pred_dnn, return_corrected_predictions=True)
    else:
        raise ValueError("Received model from unknown detection method")

    # check if labels_pred == labels; the first output will be a boolean ndarray of shape == labels.shape
    return labels_pred != labels, labels_pred


def check_adv_dnn(x, labels, model_dnn, device):
    # Get the predicted labels of the DNN and find which ones are adversarial (mis-classified)
    labels_pred = get_predicted_class(model_dnn, device, x)
    return labels_pred != labels, labels_pred


def return_indices(labels, label):
    #labels is a list
    #label is an int
    return np.where(labels == label)[0]


def get_adv_labels(x, labels, model_dnn, device, labels_uniq, runner_up_class=True):
    # Find the adversarial labels that the inputs should be mis-classified into.
    # returns a numpy array of adversarial target labels
    if runner_up_class:
        output = get_runner_up_class(model_dnn, device, x)
    else:
        max_ = labels_uniq[-1]
        output = [max_ - lab for lab in labels]

    return np.asarray(output, dtype=labels.dtype)


def check_valid_values(x_ten, name):
    if torch.isnan(x_ten).any() or torch.isinf(x_ten).any():
        raise ValueError("Nan or Inf value encountered in '{}'".format(name))


def atanh(x):
    return 0.5 * torch.log((1. + x) / (1. - x))


def sigmoid(x, a=1.):
    return torch.sigmoid(a * x)


def preprocess_input(x_embeddings, indices):
    #function that will return:
    # a dictionary `output` where `output[label]` corresponds to all input embeddings with labels == label
    output = {}
    for l, ind in indices.items():
        # Embeddings from each layer for the samples from class `l`
        output[l] = [arr[ind, :] for arr in x_embeddings]

    return output


def attack(model_dnn, device, x_orig, label_orig, labels_pred_dnn_orig, reps, labels_uniq, sigma_per_layer,
           model_detector=None, untargeted=False, dist_metric='euclidean', verbose=True, fast_mode=True):
    """
    Main function implementing the custom attack on KNN based methods.

    :param model_dnn: Trained DNN model in torch format.
    :param device: CPU or GPU device
    :param x_orig: torch tensor with a batch of clean samples for which to create attack samples. Expected to have
                   size like `(batch_size, dim1, dim2, dim3)`
    :param label_orig: torch tensor with the correct labels corresponding to samples in `x_orig`.
    :param labels_pred_dnn_orig: numpy array with the DNN-predicted labels for the batch of inputs `x_orig`.
    :param reps: dict mapping each class in `labels_uniq` to a list of torch tensors. List has length equal to the
                 number of layers and each torch tensor corresponds to the layer embeddings from the particular
                 class. These layer embeddings are used as the representative samples in the loss function.
    :param labels_uniq: list or numpy array with the distinct class labels.
    :param model_detector: Detector model that is to be attacked. Set to None in order to attack the DNN model.
    :param sigma_per_layer: Scale of the Gaussian kernel per layer for each sample in the batch. A torch tensor of
                            size `(batch_size, n_layers)`.
    :param untargeted: Set to True to run the untargeted version of the attack.
    :param dist_metric: distance metric to use. Valid values are 'euclidean' and 'cosine'.
    :param verbose: set to True to print log messages.
    :param fast_mode: set to True to run a few number of iterations and binary steps.

    :return: (x_adv, labels_pred_adv, best_dist, is_correct, mask_adver)
    """
    if fast_mode:
        binary_search_steps = 10
        max_iterations = 500
    else:
        # Not all the binary search steps will be run each time because there is a convergence check based on the
        # average interval width
        binary_search_steps = 20
        max_iterations = 1000

    thresh_bounds = 0.01
    check_adv_steps = 100
    learning_rate = 1e-2
    initial_const = 1.0
    random_start = False

    # Ensure the DNN model is in evaluation mode
    if model_dnn.training:
        model_dnn.eval()

    # shape of the input tensor
    input_shape = tuple(x_orig.size())

    if dist_metric not in ['euclidean', 'cosine']:
        raise ValueError("Specified distance metric '{}' is not supported".format(dist_metric))

    # Get the range of values in `x_orig`
    min_, max_ = get_data_bounds(x_orig, alpha=0.99)
    batch_size = x_orig.size(0)
    x_adv = x_orig.clone().detach()

    n_layers = len(reps[labels_uniq[0]])
    assert sigma_per_layer.shape == (batch_size, n_layers), "Input 'sigma_per_layer' does not have the expected shape"
    
    # label_orig is converted to ndarray
    label_orig = label_orig.detach().cpu().numpy()
    label_orig_uniq = np.unique(label_orig)
    # indices for each distinct label
    input_indices = {i: return_indices(label_orig, i) for i in label_orig_uniq}

    if not untargeted:
        # get adversarial labels i.e. the labels we want the inputs to be misclassified as
        label_adv = get_adv_labels(x_orig, label_orig, model_dnn, device, labels_uniq, runner_up_class=True)
        label_adv_uniq = np.unique(label_adv)
        # indices for each distinct label
        target_indices = {i: return_indices(label_adv, i) for i in label_adv_uniq}


    def to_attack_space(x):
        # map from [min_, max_] to [-1, +1]
        a = (min_ + max_) / 2.
        b = (max_ - min_) / 2.
        # map to the open interval (-1, 1)
        lb = -1 + 1e-16
        ub = 1 - 1e-16
        x = torch.clamp((x - a) / b, min=lb, max=ub)

        # from (-1, +1) to (-inf, +inf)
        return atanh(x)

    def to_model_space(x):
        """Transforms an input from the attack space to the model space.
        This transformation and the returned gradient are elementwise."""
        # map from (-inf, +inf) to (-1, +1)
        x = torch.tanh(x)
        # map from (-1, +1) to (min_, max_)
        a = (min_ + max_) / 2.
        b = (max_ - min_) / 2.

        return b * x + a


    # variables representing inputs in attack space will be prefixed with z
    z_orig = to_attack_space(x_orig)
    check_valid_values(z_orig, name='z_orig')
    x_recon = to_model_space(z_orig)
    check_valid_values(x_recon, name='x_recon')

    # declare tensors that keep track of constants and binary search
    # note: might not need them all
    const = torch.zeros(batch_size, device=device)
    const += initial_const
    lower_bound = torch.zeros_like(const)
    upper_bound = torch.zeros_like(const) + INFTY
    best_dist = torch.zeros_like(const) + INFTY

    # obtain layer embeddings for the clean inputs `x_orig`; gradients not needed here
    with torch.no_grad():
        x_embeddings = extract_input_embeddings(model_dnn, device, x_orig)

    # mis-classifications with clean inputs
    if model_detector is None:
        is_error, labels_pred_adv = check_adv_dnn(x_orig, label_orig, model_dnn, device)
    else:
        is_error, labels_pred_adv = check_adv_detec(x_embeddings, label_orig, labels_pred_dnn_orig, model_detector)

    is_correct = np.logical_not(is_error)
    n_correct = is_correct.sum()
    print("\n{:d} out of {:d} samples are correctly classified without any perturbation.".format(n_correct,
                                                                                                 batch_size))
    is_adv = np.zeros(batch_size, dtype=np.bool)
    mask_adver = is_adv
    # `best_dist` is set to 0 for clean inputs that are already mis-classified
    best_dist[is_error] = 0.
    if n_correct == 0:
        print("All samples from this batch are mis-classified without any perturbation. Nothing to do.")
        x_clean = x_orig.detach().cpu().numpy()

        return x_clean, labels_pred_adv, best_dist.detach().cpu().numpy(), is_correct, mask_adver

    n_reps = sum([reps[c][0].size(0) for c in labels_uniq])
    log_interval = int(np.ceil(max_iterations / 10.))
    for binary_search_step in range(binary_search_steps):
        # initialize perturbation in transformed space
        if not random_start:
            z_delta = torch.zeros_like(z_orig, requires_grad=True, device=device)
        else:
            rand = 1e-3 * np.random.randn(*input_shape)
            z_delta = torch.tensor(rand, dtype=torch.float32, requires_grad=True, device=device)

        # create a new optimizer
        optimizer = optim.RMSprop([z_delta], lr=learning_rate)

        for iteration in range(max_iterations):
            optimizer.zero_grad()
            x = to_model_space(z_orig + z_delta)
            # `requires_grad` flag should be automatically set for `x`

            # obtain embeddings for the input x
            x_embeddings = extract_input_embeddings(model_dnn, device, x)
            if not untargeted:
                loss, dist = loss_function_targeted(
                    x, x_recon, x_embeddings, reps, input_indices, target_indices, n_layers, device, const,
                    sigma_per_layer, dist_metric=dist_metric
                )
            else:
                loss, dist = loss_function_untargeted(
                    x, x_recon, x_embeddings, reps, input_indices, labels_uniq, n_reps, n_layers, device, const,
                    sigma_per_layer, dist_metric=dist_metric
                )

            # makes code slower; uncomment if necessary
            # torch.cuda.empty_cache()
            loss.backward()
            optimizer.step()
            # Note that `z_delta` is updated but `x` is not updated here
            if verbose and ((iteration % log_interval == 0) or iteration == (max_iterations - 1)):
                print('    step: %d; loss: %.6f; dist: %.4f' % (iteration, loss.item(), dist.mean().item()))

            # every `check_adv_steps` save adversarial samples with minimal perturbation
            if (iteration + 1) % check_adv_steps == 0:
                if model_detector is None:
                    is_adv, _ = check_adv_dnn(x, label_orig, model_dnn, device)
                else:
                    is_adv, _ = check_adv_detec(x_embeddings, label_orig, labels_pred_dnn_orig, model_detector)

                for i in range(batch_size):
                    if is_adv[i] and best_dist[i] > dist[i]:
                        x_adv[i] = x[i].detach()
                        best_dist[i] = dist[i].detach()

        # final `x` and its layer embeddings
        with torch.no_grad():
            x = to_model_space(z_orig + z_delta)
            # distance between the original and perturbed inputs
            dist = ((x - x_recon).view(batch_size, -1) ** 2).sum(1).sqrt()

        # check which inputs are adversarial
        if model_detector is None:
            is_adv, _ = check_adv_dnn(x, label_orig, model_dnn, device)
        else:
            with torch.no_grad():
                x_embeddings = extract_input_embeddings(model_dnn, device, x)
                is_adv, _ = check_adv_detec(x_embeddings, label_orig, labels_pred_dnn_orig, model_detector)

        # mask_adver = np.logical_and(is_adv, is_correct)
        count_bisection = 0
        for i in range(batch_size):
            if is_adv[i]:
                upper_bound[i] = const[i]
            else:
                lower_bound[i] = const[i]

            # set new const
            if upper_bound[i] >= INFTY:
                # exponential search if adv has not been found
                const[i] *= 10.
            elif lower_bound[i] <= 0:
                const[i] /= 10.
            else:
                # binary search if adv has been found
                const[i] = (lower_bound[i] + upper_bound[i]) / 2.
                if is_correct[i]:
                    count_bisection += 1

            # Update `x_adv` based on the final `x`
            if is_adv[i] and best_dist[i] > dist[i]:
                x_adv[i] = x[i].detach()
                best_dist[i] = dist[i].detach()

        # check the final attack success rate (combined with previous binary search steps)
        if model_detector is None:
            is_adv, _ = check_adv_dnn(x_adv, label_orig, model_dnn, device)
        else:
            # obtain embeddings for the inputs `x_adv`; gradients not needed here
            with torch.no_grad():
                x_embeddings = extract_input_embeddings(model_dnn, device, x_adv)
                is_adv, labels_pred_adv = check_adv_detec(x_embeddings, label_orig, labels_pred_dnn_orig,
                                                          model_detector)

        mask_adver = np.logical_and(is_adv, is_correct)
        # average (relative) difference between the upper and lower bound of the bisection interval
        diff_bounds = (upper_bound - lower_bound) / torch.clamp(lower_bound, min=1e-16)
        diff_bounds_avg = diff_bounds[is_correct].mean().item()
        if verbose:
            n_adver = mask_adver.sum()
            print('binary step: %d; num successful adv so far: %d/%d' % (binary_search_step, n_adver, batch_size))
            if n_adver:
                # average perturbation norm of adversarial samples
                norm_avg_adv = best_dist[mask_adver].mean().item()
                diff_bounds_adv_avg = diff_bounds[mask_adver].mean().item()
                if diff_bounds_adv_avg < 100.:
                    print('binary step: %d; avg_norm_adver: %.6f; avg_diff_bounds: %.4f' %
                          (binary_search_step, norm_avg_adv, diff_bounds_adv_avg))
                else:
                    print('binary step: %d; avg_norm_adver: %.6f' % (binary_search_step, norm_avg_adv))

        if diff_bounds_avg < thresh_bounds:
            print("Exiting bisection search early based on the average interval width")
            break

    if verbose and count_bisection < n_correct:
        print("\n{:d} out of {:d} samples did not enter the bisection search phase".
              format(n_correct - count_bisection, n_correct))

    check_valid_values(x_adv, name='x_adv')

    # Returning the adversarial inputs and corresponding predicted labels as numpy arrays.
    # Also returning the norm of the adversarial perturbations, and boolean arrays indicating which samples were
    # correctly classified and which ones are adversarial.
    return x_adv.detach().cpu().numpy(), labels_pred_adv, best_dist.detach().cpu().numpy(), is_correct, mask_adver
