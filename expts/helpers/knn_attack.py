import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import scipy
import os
from numpy import linalg as LA
from torch.distributions.multivariate_normal import MultivariateNormal
from helpers.constants import ROOT
from helpers.utils import combine_and_vectorize, get_data_bounds

INFTY = 1e20


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


def extract_layer_embeddings(model, device, data_loader, indices):
    # returns a list of torch tensors, where each torch tensor is a per layer embedding
    if model.training:
        model.eval()

    output = {}
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

    for i in range(n_layers):
        # Combine all the data batches and convert the numpy array to torch tensor
        embeddings_comb = combine_and_vectorize(embeddings[i])

        for label in indices.keys():
            temp_arr = torch.from_numpy(embeddings_comb[indices[label], :]).to(device)
            if i == 0:
                output[label] = [temp_arr]
            else:
                output[label].append(temp_arr)

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

        # TODO: Check if `tens` has its `requires_grad` set because it is derived from `x_input`
        # tens.requires_grad = True
        outputs_layers[i] = tens

    return outputs_layers


def sum_gaussian_kernels(x, reps, sigma, metric='euclidean'):
    """
    Compute the sum of Gaussian kernels evaluated at the set of representative points `reps` and centered on `x`.

    :param x: torch tensor with a particular layer embedding of the perturbed inputs. Specifically, `f_l(x + delta)`.
              It is expected to be a tensor of shape `(n_samp, n_dim)`, where `n_samp` is the number of inputs and
              `n_dim` is the number of dimensions.
    :param reps: torch tensor with the layer embeddings of the representative samples from a particular class.
                 Specifically, `f_l(x_n)` for the set of `n` from a given class. Expected to be a tensor of shape
                 `(n_reps, n_dim)`, where `n_reps` is the number of representative samples and `n_dim` is the number
                 of dimensions.
    :param sigma: scale or standard deviation of the Gaussian kernel for layer `l`.
    :param metric: distance metric. Set to 'euclidean' or 'cosine'.

    :return: torch tensor of shape `(n_samp, )` with the sum of the Gaussian kernels for each of the `n_samp`
             inputs in `x`.
    """
    n_samp, d = x.size()
    n_reps, n_dim = reps.size()
    assert d == n_dim, "Mismatch in the input dimensions"

    if metric == 'cosine':
        # Rescale the vectors to unit norm, which makes the euclidean distance equal to the cosine distance (times
        # a scale factor)
        norm_x = torch.norm(x, p=2, dim=1) + 1e-16
        x_n = torch.div(x, norm_x.view(n_samp, 1))
        norm_reps = torch.norm(reps, p=2, dim=1) + 1e-16
        reps_n = torch.div(reps, norm_reps.view(n_reps, 1))
    elif metric == 'euclidean':
        x_n = x
        reps_n = reps
    else:
        raise ValueError("Invalid value '{}' for the input 'metric'".format(metric))

    # Compute pairwise euclidean distances
    dist_mat = torch.squeeze(torch.cdist(torch.unsqueeze(x_n, 0), torch.unsqueeze(reps_n, 0), p=2), 0)
    temp_ten = (-1. / (sigma * sigma)) * torch.pow(dist_mat, 2)
    # `dist_mat` and `temp_ten` will have the same size `(n_samp, n_reps)`
    max_val, _ = torch.max(temp_ten, dim=1)
    # `max_val` will have size `(n_samp, )`

    return torch.exp(max_val) * torch.exp(temp_ten - torch.unsqueeze(max_val, 1)).sum(1)


def loss_function(x, x_recon, x_embeddings, reps, input_indices, n_layers, device, const, sigma=1.0):

    batch_size = x.size(0)
    max_label = max(reps.keys())
    # double summation based on the paper
    adv_loss = torch.zeros(batch_size, device=device)
    for i in range(n_layers):
        for c, ind_c in input_indices.items():
            # TODO: `c_hat` needs to be changed after testing
            c_hat = max_label - c
            # input1 = x_embeddings[i][ind_c, :]
            # input2 = reps[c][i]
            # print("reps1", reps[c][i].requires_grad)

            # embeddings from layer `i` for the samples from class `c`
            adv_loss[ind_c] = (adv_loss[ind_c]
                               + sum_gaussian_kernels(x_embeddings[i][ind_c, :], reps[c][i], sigma)
                               - sum_gaussian_kernels(x_embeddings[i][ind_c, :], reps[c_hat][i], sigma))

    # obtaining the distance between the input (with noise added) and the original input (with no noise)
    dist = ((x - x_recon).view(batch_size, -1) ** 2).sum(1)

    # `const` multiplies the loss term to be consistent with CW attack formulation.
    # Smaller values of `const` lead to solutions with smaller perturbation norm
    total_loss = dist.to(device) + const * adv_loss

    # returned output
    return total_loss.mean(), dist.sqrt()
    

def check_adv(x_embeddings, labels, det_model):
    # `x_embeddings` will be a list of torch tensors. It needs to be converted into a list of numpy arrays before
    # calling the deep KNN score method
    x_embeddings_np = [tens.detach().cpu().numpy() for tens in x_embeddings]

    # scores and class predictions from the detection model
    _, labels_pred = det_model.score(x_embeddings_np)

    # check if labels_pred == labels; the output will be a boolean ndarray of shape == labels.shape
    return labels_pred != labels


def return_indices(labels, label):
    #labels is a list
    #label is an int
    return np.where(labels == label)[0]


def get_adv_labels(labels, labels_uniq):
    # Given an ndarray `labels` and a list of unique labels, return an ndarray of adv. labels i.e. the labels the
    # input should be misclassified as
    max_ = labels_uniq[-1]
    output = []
    for element in labels:
        output.append(max_ - element)

    output = np.asarray(output, dtype=labels.dtype)
    return np.reshape(output, labels.shape)


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


#below is the main attack function
#pending verification E2E
def attack(model, device, data_loader, x_orig, label_orig, dknn_model):
    # x_orig is a torch tensor
    # label_orig is a torch tensor
    
    init_mode = 1
    init_mode_k = 1
    binary_search_steps = 1
    max_iterations = 5
    learning_rate = 1e-2
    initial_const = 1.
    max_linf = None
    random_start = False
    thres_steps = 100
    check_adv_steps = 100
    verbose = True

    # Ensure the DNN model is in evaluation mode
    if model.training:
        model.eval()
        
    #all labels of the data_loader + range of labels
    labels, labels_uniq = get_labels(data_loader)
    
    # Get the range of values in `x_orig`
    min_, max_ = get_data_bounds(x_orig, alpha=0.99)
    # min_ = torch.tensor(0., device=device)
    # max_ = torch.tensor(1., device=device)
    if max_linf is not None:
        min_ = torch.max(x_orig - max_linf, min_)
        max_ = torch.min(x_orig + max_linf, max_)

    batch_size = x_orig.size(0)
    x_adv = x_orig.clone()
    
    # label_orig is converted to ndarray
    label_orig = label_orig.detach().cpu().numpy()
    label_orig_uniq = np.unique(label_orig)

    # get adversarial labels i.e. the labels we want the inputs to be misclassified as
    label_adv = get_adv_labels(label_orig, label_orig_uniq)
    assert type(label_orig) == type(label_adv)
    assert label_orig.shape == label_adv.shape

    # shape of the input tensor
    input_shape = tuple(x_orig.size())
    
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
    #note: might not need them all
    const = torch.zeros(batch_size, device=device)
    const += initial_const
    lower_bound = torch.zeros_like(const)
    upper_bound = torch.zeros_like(const) + INFTY
    best_dist = torch.zeros_like(const) + INFTY

    # contains the indices of the dataset corresponding to a particular label i.e. indices[i] contains the indices
    # of all elements whose label is i
    indices = {i: return_indices(labels, i) for i in labels_uniq}

    # indices for each distinct label in `label_orig`. Needs to be computed only once
    input_indices = {i: return_indices(label_orig, i) for i in label_orig_uniq}

    target_indices = {}     # not used
    
    # `reps` contains the layer wise embeddings for the entire dataloader
    # gradients are not required for the representative samples
    reps = extract_layer_embeddings(model, device, data_loader, indices)
    n_layers = len(reps[labels_uniq[0]])
    
    for binary_search_step in range(binary_search_steps):
        # initialize perturbation in transformed space
        if not random_start:
            z_delta = torch.zeros_like(z_orig, requires_grad=True, device=device)
        else:
            rand = 1e-2 * np.random.randn(*input_shape)
            z_delta = torch.tensor(rand, dtype=torch.float32, requires_grad=True, device=device)

        # create a new optimizer
        optimizer = optim.RMSprop([z_delta], lr=learning_rate)

        for iteration in range(max_iterations):
            optimizer.zero_grad()
            x = to_model_space(z_orig + z_delta)
            # `requires_grad` flag should be automatically set for `x`

            # obtain embeddings for the input x
            x_embeddings = extract_input_embeddings(model, device, x)
            # classwise_x = preprocess_input(x_embeddings, input_indices)
            
            loss, dist = loss_function(x, x_recon, x_embeddings, reps, input_indices, n_layers, device, const)
            torch.cuda.empty_cache() #added here
            loss.backward()
            optimizer.step()

            if verbose and iteration % (np.ceil(max_iterations / 10)) == 0:
                print('    step: %d; loss: %.6f; dist: %.4f' % (iteration, loss.item(), dist.mean().item()))

            # every <check_adv_steps>, save adversarial samples
            # with minimal perturbation
            if ((iteration + 1) % check_adv_steps) == 0 or iteration == max_iterations:
                is_adv = check_adv(x_embeddings, label_orig, dknn_model)
                for i in range(batch_size):
                    if is_adv[i] and best_dist[i] > dist[i]:
                        x_adv[i] = x[i]
                        best_dist[i] = dist[i]

        # check how many attacks have succeeded
        with torch.no_grad():
            is_adv = check_adv(x_embeddings, label_orig, dknn_model)
            if verbose:
                print('binary step: %d; num successful adv: %d/%d' % (binary_search_step, is_adv.sum(), batch_size))

        for i in range(batch_size):
            # only keep adv with smallest l2dist
            if is_adv[i] and best_dist[i] > dist[i]:
                x_adv[i] = x[i]
                best_dist[i] = dist[i]

        # check the final attack success rate (combined with previous binary search steps)
        if verbose:
            # obtain embeddings for the inputs `x_adv`
            x_embeddings = extract_input_embeddings(model, device, x_adv)
            with torch.no_grad():
                is_adv = check_adv(x_embeddings, label_orig, dknn_model)
                print('binary step: %d; num successful adv so far: %d/%d' % (binary_search_step, is_adv.sum(),
                                                                             batch_size))

    check_valid_values(x_adv, name='x_adv')
    return x_adv
