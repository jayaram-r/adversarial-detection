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
from helpers.utils import combine_and_vectorize

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

    return labels, [labels_uniq[0], labels_uniq[-1]]

def extract_layer_embeddings(model, device, data_loader, indices, requires_grad=True):
    # returns a list of torch tensors, where each torch tensor is a per layer embedding
    if model.training:
        model.eval()

    output = {}
    labels = list(indices.keys())

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

    for label in labels:
        indices_needed = list(indices[label])
        for i in range(n_layers):
            # Combine all the data batches and convert the numpy array to torch tensor
            temp_arr = combine_and_vectorize(embeddings[i])
            temp_arr = temp_arr[indices_needed]
            temp_arr = torch.from_numpy(temp_arr).to(device)
            temp_arr.requires_grad = requires_grad
            if i == 0:
                output[label] = [temp_arr]
            else:
                output[label].append(temp_arr)
            #embeddings[i] = temp_arr
            #embeddings[i] = torch.from_numpy(temp_arr).to(device)
            #embeddings[i].requires_grad = requires_grad
    
    return output


def extract_input_embeddings(model, device, x_input, requires_grad=True):
    # Returns a list of torch tensors, where each torch tensor is a per layer embedding
    # Model should already be set to eval mode
    # if model.training:
    #    model.eval()

    with torch.no_grad():
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

        tens.requires_grad = requires_grad
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

    # `x` is a function of the perturbation vector, which is the variable of optimization
    #x.requires_grad = True

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
    dist_mat = torch.cdist(torch.unsqueeze(x_n, 0), torch.unsqueeze(reps_n, 0), p=2)
    dist_mat = torch.squeeze(dist_mat, 0)
    temp_ten = (-1. / (sigma * sigma)) * torch.pow(dist_mat, 2)
    # `dist_mat` and `temp_ten` will have the same size `(n_samp, n_reps)`
    max_val, _ = torch.max(temp_ten, dim=1)
    # `max_val` will have size `(n_samp, )`

    return torch.exp(max_val) * torch.exp(temp_ten - torch.unsqueeze(max_val, 1)).sum(1)


#not useful?
def get_distance(p1, p2, dist_type='cosine'):
    #returns either the cosine distance or the euclidean distance, depending on what is specified
    numerator = torch.dot(p1.view(-1,),p2.view(-1,))
    p1_norm = torch.norm(p1, 2)
    p2_norm = torch.norm(p2, 2)
    denominator = torch.mul(p1_norm, p2_norm)
    val = torch.div(numerator, denominator)
    #print(numerator, denominator, val)
    return val

def loss_function(x, x_recon, x_embeddings, reps, device, indices, const, label, min_label, max_label):

    batch_size = x.size(0)
    num_embeddings = len(reps[0])

    def preprocess_input(x_embeddings, label, min_label, max_label):
        indices = {}
        output = {}
        num_embeddings = len(x_embeddings)
        for l in range(min_label, max_label+1):
            indices[l] = np.where(label == l)[0]
           
        for i in range(num_embeddings):
            for l in range(min_label, max_label+1):
                if i == 0:
                    output[l] = [x_embeddings[i][list(indices[l])]]
                else:
                    output[l].append(x_embeddings[i][list(indices[l])])

        return output

    classwise_x = preprocess_input(x_embeddings, label, min_label, max_label)
    
    uniq_labels = [k for k in classwise_x.keys()]

    for i in range(num_embeddings):
        for uniq_label in uniq_labels:
            input1 = classwise_x[uniq_label][i]
            input2 = reps[uniq_label][i]
            print(input1)
            print(input2)
            exit()
            val = sum_gaussian_kernels(input1, input2, 1)
            print(val, val.sum(0))

    return None
    
#this function below needs to be re-written to factor in prediction from knn and not model
#pending
def check_adv(x, label, model, device):
    y_pred = model(x).argmax(1)
    return torch.tensor((y_pred != label).astype(np.float32)).to(device)


def return_indices(labels, label):
    #labels is a list
    #label is an int
    return np.where(labels == label)[0]


def get_adv_labels(label, label_range):
    #given an ndarray `label` and a list of label ranges, return an ndarray of adv. labels i.e. the labels the input should be misclassified as 
    min_ = label_range[0]
    max_ = label_range[1]
    label_shape = label.shape
    label = list(label)
    output = []
    for element in label:
        val = max_ - element
        output.append(val)
    output = np.asarray(output)
    output = np.reshape(output, label_shape)
    return output


def atanh(x):
    return 0.5 * torch.log((1. + x) / (1. - x))


def sigmoid(x, a=1.):
    return torch.sigmoid(a * x)


#below is the main attack function
#pending
def attack(model, device, data_loader, x_orig, label):
    # x_orig is a torch tensor
    # label is a torch tensor
    
    init_mode = 1
    init_mode_k = 1
    binary_search_steps = 1
    max_iterations = 500
    learning_rate = 1e-2
    initial_const = 1.
    max_linf = None
    random_start = False
    thres_steps = 100
    check_adv_steps = 100
    verbose = True
        
    #all labels of the data_loader + range of labels 
    labels, labels_range = get_labels(data_loader)
    min_label = labels_range[0]
    max_label = labels_range[1]

    min_, max_ = x_orig.min(), x_orig.max()
    # min_ = torch.tensor(0., device=device)
    # max_ = torch.tensor(1., device=device)
    if max_linf is not None:
        min_ = torch.max(x_orig - max_linf, min_)
        max_ = torch.min(x_orig + max_linf, max_)

    batch_size = x_orig.size(0)
    x_adv = x_orig.clone()
    
    #label is converted to ndarray
    label = label.cpu().numpy()

    #get adversarial labels i.e. the labels we want the inputs to be misclassified as 
    adv_label = get_adv_labels(label, labels_range)

    #verify if type and shape of adv_labels is the same as label
    assert type(label) == type(adv_label)
    assert label.shape == adv_label.shape

    #shape of the input tensor, which is now converted to ndarray
    input_shape = x_orig.detach().cpu().numpy().shape
    
    def to_attack_space(x):
        # map from [min_, max_] to [-1, +1]
        a = (min_ + max_) / 2.
        b = (max_ - min_) / 2.
        # map to the open interval (-1, 1)
        x = (1 - 1e-16) * ((x - a) / b)

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
    x_recon = to_model_space(z_orig)

    # declare tensors that keep track of constants and binary search
    #note: might not need them all
    const = torch.zeros((batch_size, ), device=device)
    const += initial_const
    lower_bound = torch.zeros_like(const)
    upper_bound = torch.zeros_like(const) + INFTY
    best_dist = torch.zeros_like(const) + INFTY


    # contains the indices of the dataset corresponding to a particular label i.e. indices[i] contains the indices
    # of all elements whose label is i
    indices = {i: return_indices(labels, i) for i in range(min_label, max_label + 1)}
    target_indices = {}
    
    # `reps` contains the layer wise embeddings for the entire dataloader
    # gradients are not required for the representative samples
    reps = extract_layer_embeddings(model, device, data_loader, indices, requires_grad=False)
    
    for binary_search_step in range(binary_search_steps):

        # initialize perturbation in transformed space
        if not random_start:
            z_delta = torch.zeros_like(z_orig, requires_grad=True)
        else:
            rand = np.random.randn(*input_shape) * 1e-2
            z_delta = torch.tensor(rand, dtype=torch.float32, requires_grad=True, device=device)

        # create a new optimizer
        optimizer = optim.RMSprop([z_delta], lr=learning_rate)

        for iteration in range(max_iterations):
            optimizer.zero_grad()
            x = to_model_space(z_orig + z_delta)

            #obtain embeddings for the input x
            x_embeddings = extract_input_embeddings(model, device, x)
            
            loss = loss_function(x, x_recon, x_embeddings, reps, device, indices, const, label, min_label, max_label)
            exit()
            loss.backward()
            optimizer.step()

            if verbose and iteration % (np.ceil(max_iterations / 10)) == 0:
                print('    step: %d; loss: %.3f; dist: %.3f' % (iteration, loss.cpu().detach().numpy(), dist.mean().cpu().detach().numpy()))

            # every <check_adv_steps>, save adversarial samples
            # with minimal perturbation
            if ((iteration + 1) % check_adv_steps) == 0 or iteration == max_iterations:
                is_adv = check_adv(x, label)
                for i in range(batch_size):
                    if is_adv[i] and best_dist[i] > dist[i]:
                        x_adv[i] = x[i]
                        best_dist[i] = dist[i]

        # check how many attacks have succeeded
        with torch.no_grad():
            is_adv = check_adv(x, label)
            if verbose:
                print('binary step: %d; num successful adv: %d/%d' % (binary_search_step, is_adv.sum(), batch_size))

        for i in range(batch_size):
            # only keep adv with smallest l2dist
            if is_adv[i] and best_dist[i] > dist[i]:
                x_adv[i] = x[i]
                best_dist[i] = dist[i]

        # check the current attack success rate (combined with previous
        # binary search steps)
        if verbose:
            with torch.no_grad():
                is_adv = check_adv(x_adv, label)
                print('binary step: %d; num successful adv so far: %d/%d' %(binary_search_step, is_adv.sum(), batch_size))

    return x_adv
