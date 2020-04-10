import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import scipy
import os
from numpy import linalg as LA
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


'''
def make_tensor(e1, e2, requires_grad=True):
    #unused
    with torch.set_grad_enabled(requires_grad):
        n = len(e1) #should be equal to the batch size
        m = len(e1[0]) #should be equal to the number of layers from which we extract embeddings
        output1 = []
        for i in range(0, m):
            for j in range(0, n):
                if j == 0:
                    temp = e1[j][i]
                else:
                    temp = np.vstack((temp, e1[j][i]))
            output1.append(torch.from_numpy(temp))

        output2 = []
        n = len(e2)
        m = len(e2[0])
        for i in range(0, m):
            for j in range(0, n):
                if j == 0:
                    temp = e2[j][i]
                else:
                    temp = np.vstack((temp, e2[j][i]))
            output2.append(torch.from_numpy(temp))

    return output1, output2
'''


def extract_layer_embeddings(model, device, data_loader, requires_grad=True):
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

    for i in range(n_layers):
        # Combine all the data batches and convert the numpy array to torch tensor
        temp_arr = combine_and_vectorize(embeddings[i])
        embeddings[i] = torch.from_numpy(temp_arr).to(device)
        embeddings[i].requires_grad = requires_grad

    return embeddings


def extract_input_embeddings(model, device, x_input, requires_grad=True):
    #returns a list of torch tensors, where each torch tensor is a per layer embedding
    if model.training:
        model.eval()

    with torch.no_grad():
        data = x_input.to(device)
        # Layer outputs
        outputs_layers = model.layer_wise(data)

    embeddings = []
    for v in outputs_layers:
        # Flatten all but the first dimension of the tensor and set/unset its requires gradient flag
        v = v.view(v.size(0), -1)
        v.requires_grad = requires_grad
        embeddings.append(v)

    return embeddings


def get_distance(p1, p2, dist_type='cosine'):
    #returns either the cosine distance or the euclidean distance, depending on what is specified
    numerator = torch.dot(p1.view(-1,),p2.view(-1,))
    p1_norm = torch.norm(p1, 2)
    p2_norm = torch.norm(p2, 2)
    denominator = torch.mul(p1_norm, p2_norm)
    val = torch.div(numerator, denominator)
    #print(numerator, denominator, val)
    return val


def loss_function(x, x_recon, x_embedding, reps, device, indices, const, label, max_label):
    #loss function needed by the attack formulation
    #pending
    
    def gaussian_kernel(dist):
        inp = torch.mul(dist, dist)
        val = torch.exp(inp)
        return val

    src_indices = indices
    target_indices = indices

    batch_size = x.size(0)
    #embeddings, labels, labels_pred, counts = extract_layer_embeddings(model, device, data_loader)
    num_embeddings = len(reps)
    adv_loss = torch.zeros((batch_size))
    #, device=device)

    num_embeddings = len(reps)
    
    #must find way to avoid using loops
    for embedding_num in range(0, num_embeddings):
        rep = reps[embedding_num] #is a torch tensor 
        
        for sample_num in range(0, batch_size):
            x_e = x_embedding[embedding_num][sample_num]
            x_label = label[sample_num]

            #obtain those indices of the dataset corresponding to label == x_label from the dict `indices`
            required_src_indices = src_indices[x_label]
            required_src_indices = list(required_src_indices)

            #obtain the embeddings from the entire data_loader corresponding to the indices obtained above; desired_reps is a torch tensor
            src_reps = rep[required_src_indices] #might not be able to do this over a torch tensor
            num_reps_per_label = src_reps.size(0)
            
            temp_sum_1 = torch.Tensor([0])
            
            for k in range(0, num_reps_per_label):
                dist = get_distance(x_e, src_reps[k])
                gauss_dist = gaussian_kernel(dist)
                temp_sum_1 += gauss_dist
            
            if sample_num == 0:
                final_1 = temp_sum_1
            else:
                final_1 = torch.cat((final_1, temp_sum_1), 0)    
                
            #obtain those indices of the dataset corresponding to label == x_label from the dict `indices`
            required_target_indices = target_indices[max_label - x_label]
            required_target_indices = list(required_target_indices)
            
            #obtain the embeddings from the entire data_loader corresponding to the indices obtained above; desired_reps is a torch tensor
            target_reps = rep[required_target_indices] #might not be able to do this over a torch tensor
            num_reps_per_label = target_reps.size(0)

            temp_sum_2 = torch.Tensor([0])
            
            for k in range(0, num_reps_per_label):
                dist = get_distance(x_e, target_reps[k])
                gauss_dist = gaussian_kernel(dist)
                temp_sum_2 += gauss_dist

            if sample_num == 0:
                final_2 = temp_sum_2
            else:
                final_2 = torch.cat((final_2, temp_sum_2), 0)
        
        diff = torch.sub(final_1, final_2)
        #diff.to(device)
        #print(final_1, final_2)
        #exit()
        adv_loss = torch.add(adv_loss, diff)
        print(adv_loss, adv_loss.shape)
    return adv_loss


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
    return 0.5 * torch.log((1 + x) / (1 - x))


def sigmoid(x, a=1):
    return 1 / (1 + torch.exp(-a * x))


#below is the main attack function
#pending
def attack(model, device, data_loader, x_orig, label):

    #x_orig is a torch tensor
    #label is a torch tensor
    
    init_mode=1 
    init_mode_k=1 
    binary_search_steps=1
    max_iterations=500 
    learning_rate=1e-2 
    initial_const=1
    max_linf=None
    random_start=False 
    thres_steps=100
    check_adv_steps=100
    verbose=True
        
    #all labels of the data_loader + range of labels 
    labels, labels_range = get_labels(data_loader)
    
    min_label = labels_range[0]
    max_label = labels_range[1]

    min_ = torch.tensor(0., device=device)
    max_ = torch.tensor(1., device=device)
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
        a = (min_ + max_) / 2
        b = (max_ - min_) / 2
        x = (x - a) / b
        # from [-1, +1] to approx. (-1, +1)
        x = x * 0.999999
        # from (-1, +1) to (-inf, +inf)
        return atanh(x)

    def to_model_space(x):
        """Transforms an input from the attack space to the model space.
        This transformation and the returned gradient are elementwise."""
        # from (-inf, +inf) to (-1, +1)
        x = torch.tanh(x)
        # map from (-1, +1) to (min_, max_)
        a = (min_ + max_) / 2
        b = (max_ - min_) / 2
        x = x * b + a
        return x

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


    #contains the indices of the dataset corresponding to a particular label i.e. indices[i] contains the indices of all elements whose label is i
    indices = {}
    target_indices = {}
    for i in range(min_label, max_label + 1):
        indices[i] = return_indices(labels, i)
    

    #reps contains the layer wise embeddings for the entire dataloader
    reps = extract_layer_embeddings(model, device, data_loader)

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
            
            loss = loss_function(x, x_recon, x_embeddings, reps, device, indices, const, label, max_label)
            exit()
            loss.backward()
            optimizer.step()

            if (verbose and iteration % (np.ceil(max_iterations / 10)) == 0):
                print('    step: %d; loss: %.3f; dist: %.3f' %(iteration, loss.cpu().detach().numpy(), dist.mean().cpu().detach().numpy()))

            # every <check_adv_steps>, save adversarial samples
            # with minimal perturbation
            if ((iteration + 1) % check_adv_steps == 0 or iteration == max_iterations):
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
