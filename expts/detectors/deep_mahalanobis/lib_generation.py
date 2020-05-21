"""
Deep Mahalanobis detection method repurposed from the author's implementation:
https://github.com/pokaxpoka/deep_Mahalanobis_detector

"""
from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist, cdist, squareform
from helpers.constants import NORMALIZE_IMAGES


# lid of a batch of query points X
def mle_batch(data, batch, k):
    '''
    commpute lid score using data & batch with k-neighbors
    return: a: computed LID score
    '''
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data)-1)
    f = lambda v: - k / np.sum(np.log(v/v[-1]))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)

    return a


# this function is from https://github.com/xingjunm/lid_adversarial_subspace_detection
def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact and generate labels
    return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y


def sample_estimator(model, device, num_classes, layer_dimension_reduced, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance
    if model.training:
        model.eval()

    correct, total = 0, 0
    num_output = len(layer_dimension_reduced)   # number of layers
    num_sample_per_class = np.zeros(num_classes, dtype=np.int)
    # `list_features` is a list of length equal to the number of layers; each item is a list of 0s of length
    # number of classes
    list_features = []
    for _ in range(num_output):
        list_features.append([0] * num_classes)

    with torch.no_grad():
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            n_batch = data.size(0)
            total += n_batch

            # Get the intermediate layer embeddings and the DNN output
            output, out_features = model.layer_wise_deep_mahalanobis(data)
            # Dimension reduction for the layer embeddings.
            # Each `N x C x H x W` tensor is converted to a `N x C` tensor by average pooling
            for i in range(num_output):
                sz = out_features[i].size()
                if len(sz) > 2:
                    out_features[i] = out_features[i].view(sz[0], sz[1], -1)
                    out_features[i] = torch.mean(out_features[i], 2)
                    # print("la:", out_features[i].shape)

            # compute the accuracy
            pred = output.max(1)[1]
            correct += pred.eq(target).sum().item()

            # construct the sample matrix for each layer and each class
            for i in range(n_batch):
                label = target[i]
                if num_sample_per_class[label] == 0:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] = out[i].view(1, -1)
                        out_count += 1
                else:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] = \
                            torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                        out_count += 1

                num_sample_per_class[label] += 1

    # `list_features` will be a list of list of torch tensors. The first list indexes the layers and the second list
    # indexes the classes. Each tensor has samples from a particular layer and a particular class
    '''
    for i in range(num_output):
        for j in range(num_classes):
            print(i, j, list_features[i][j].shape)
    '''

    # Compute the sample mean for each layer and each class
    sample_class_mean = []
    out_count = 0
    for num_feature in layer_dimension_reduced:
        temp_list = torch.zeros(num_classes, num_feature).to(device)
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)

        sample_class_mean.append(temp_list)
        out_count += 1

    '''
    print("sample_class_mean")
    for i in range(num_output):
        for j in range(num_classes):
            print(i, j, sample_class_mean[i][j].shape)
    '''

    # Sample inverse covariance matrix estimation for each layer with data from all the classes combined
    # (i.e. a shared inverse covariance matrix)
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    precision = []
    for k in range(num_output):
        X = list_features[k][0] - sample_class_mean[k][0]
        for i in range(1, num_classes):
            X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

        group_lasso.fit(X.detach().cpu().numpy())
        temp_precision = torch.from_numpy(group_lasso.precision_).to(dtype=torch.float, device=device)
        precision.append(temp_precision)

    # `precision` will be a list of torch tensors with the precision matrix per layer
    '''
    print("precision")
    for i in range(num_output):
        print(i, precision[i].shape)
    '''
    print('\n Training Accuracy:({:.4f}%)\n'.format(100. * correct / total))
    return sample_class_mean, precision


def get_Mahalanobis_score(model, device, test_loader, num_classes, outf, out_flag, net_type, sample_mean, precision,
                          layer_index, magnitude):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    if model.training:
        model.eval()

    scale_images = NORMALIZE_IMAGES[net_type][1]
    n_channels = len(scale_images)
    Mahalanobis = []
    if out_flag:
        temp_file_name = '%s/confidence_Ga%s_In.txt'%(outf, str(layer_index))
    else:
        temp_file_name = '%s/confidence_Ga%s_Out.txt'%(outf, str(layer_index))
        
    fp = open(temp_file_name, 'w')
    
    for data, target in test_loader:
        data = data.to(device=device, dtype=torch.float)
        data.requires_grad = True
        # target = target.to(device=device)
        
        out_features = model.intermediate_forward(data, layer_index)
        sz = out_features.size()
        if len(sz) > 2:
            # Dimension reduction for the layer embedding
            # `N x C x H x W` tensor is converted to a `N x C` tensor by average pooling
            out_features = out_features.view(sz[0], sz[1], -1)
            out_features = torch.mean(out_features, 2)

        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features - batch_sample_mean
            #check if both parameters to multiplication are the same type
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)
        
        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - batch_sample_mean
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
         
        gradient =  torch.ge(data.grad, 0)
        gradient = (gradient.float() - 0.5) * 2
        '''
        if net_type == 'densenet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), 
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), 
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), 
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
        elif net_type == 'resnet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), 
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), 
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), 
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
        '''
        
        if n_channels == 1:
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(device=device),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda(device=device)) / scale_images[0])
        else:
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(device=device),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda(device=device)) / scale_images[0])
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(device=device),
                                 gradient.index_select(1, torch.LongTensor([1]).cuda(device=device)) / scale_images[1])
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(device=device),
                                 gradient.index_select(1, torch.LongTensor([2]).cuda(device=device)) / scale_images[2])

        with torch.no_grad():
            tempInputs = torch.add(data, -magnitude, gradient).to(device=device, dtype=torch.float)
            noise_out_features = model.intermediate_forward(tempInputs, layer_index)
            sz = noise_out_features.size()
            if len(sz) > 2:
                noise_out_features = noise_out_features.view(sz[0], sz[1], -1)
                noise_out_features = torch.mean(noise_out_features, 2)

        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1, 1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(noise_gaussian_score.detach().cpu().numpy())
        for i in range(data.size(0)):
            fp.write("{}\n".format(noise_gaussian_score[i]))

    fp.close()
    return Mahalanobis


def get_Mahalanobis_score_adv(model, device, test_data, num_classes, net_type, sample_mean, precision,
                              layer_index, magnitude):
    '''
    Compute the proposed Mahalanobis confidence score on adversarial samples
    return: Mahalanobis score from layer_index
    '''
    if model.training:
        model.eval()

    scale_images = NORMALIZE_IMAGES[net_type][1]
    n_channels = len(scale_images)
    n_samp = test_data.size(0)
    batch_size = 100
    total = 0
    Mahalanobis = []
    while total < n_samp:
        data = test_data[total : total + batch_size].to(device=device, dtype=torch.float)
        data.requires_grad = True
        total += batch_size

        # get the intermediate layer embedding
        out_features = model.intermediate_forward(data, layer_index)
        sz = out_features.size()
        if len(sz) > 2:
            # Dimension reduction for the layer embedding
            # `N x C x H x W` tensor is converted to a `N x C` tensor by average pooling
            out_features = out_features.view(sz[0], sz[1], -1)
            out_features = torch.mean(out_features, 2)
        
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)
        
        # Input_processing
        # Class corresponding to the minimum mahalanobis distance
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - batch_sample_mean
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
         
        gradient =  torch.ge(data.grad, 0)
        gradient = (gradient.float() - 0.5) * 2
        '''
        if net_type == 'densenet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), 
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), 
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), 
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
        elif net_type == 'resnet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), 
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), 
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
        '''

        if n_channels == 1:
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(device=device),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda(device=device)) / scale_images[0])
        else:
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(device=device),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda(device=device)) / scale_images[0])
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(device=device),
                                 gradient.index_select(1, torch.LongTensor([1]).cuda(device=device)) / scale_images[1])
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(device=device),
                                 gradient.index_select(1, torch.LongTensor([2]).cuda(device=device)) / scale_images[2])
        
        with torch.no_grad():
            tempInputs = torch.add(data, -magnitude, gradient).to(device=device, dtype=torch.float)
            noise_out_features = model.intermediate_forward(tempInputs, layer_index)
            sz = noise_out_features.size()
            if len(sz) > 2:
                noise_out_features = noise_out_features.view(sz[0], sz[1], -1)
                noise_out_features = torch.mean(noise_out_features, 2)

        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1, 1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(noise_gaussian_score.detach().cpu().numpy())
        
    return Mahalanobis


# Not used
def get_posterior(model, device, net_type, test_loader, magnitude, temperature, outf, out_flag):
    '''
    Compute the maximum value of (processed) posterior distribution - ODIN
    return: null
    '''
    criterion = nn.CrossEntropyLoss()
    if model.training:
        model.eval()

    scale_images = NORMALIZE_IMAGES[net_type][1]
    n_channels = len(scale_images)
    total = 0
    if out_flag:
        temp_file_name_val = '%s/confidence_PoV_In.txt' % (outf)
        temp_file_name_test = '%s/confidence_PoT_In.txt' % (outf)
    else:
        temp_file_name_val = '%s/confidence_PoV_Out.txt' % (outf)
        temp_file_name_test = '%s/confidence_PoT_Out.txt' % (outf)

    g = open(temp_file_name_val, 'w')
    f = open(temp_file_name_test, 'w')

    for data, _ in test_loader:
        total += data.size(0)
        data = data.cuda(device=device)
        data.requires_grad = True
        batch_output = model(data)

        # temperature scaling
        outputs = batch_output / temperature
        labels = outputs.max(1)[1]
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(data.grad, 0)
        gradient = (gradient.float() - 0.5) * 2
        '''
        if net_type == 'densenet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), 
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), 
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), 
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
        elif net_type == 'resnet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), 
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), 
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), 
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
        '''

        if n_channels == 1:
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(device=device),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda(device=device)) / scale_images[0])
        else:
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(device=device),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda(device=device)) / scale_images[0])
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(device=device),
                                 gradient.index_select(1, torch.LongTensor([1]).cuda(device=device)) / scale_images[1])
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(device=device),
                                 gradient.index_select(1, torch.LongTensor([2]).cuda(device=device)) / scale_images[2])

        with torch.no_grad():
            tempInputs = torch.add(data, -magnitude, gradient)
            outputs = model(tempInputs)
            outputs = outputs / temperature
            soft_out = F.softmax(outputs, dim=1)
            soft_out, _ = torch.max(soft_out, dim=1)

        for i in range(data.size(0)):
            if total <= 1000:
                g.write("{}\n".format(soft_out[i]))
            else:
                f.write("{}\n".format(soft_out[i]))

    f.close()
    g.close()
