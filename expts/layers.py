from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from nets.mnist import *
from nets.cifar10 import *
from nets.svhn import *
import os
import foolbox
import sys
from pympler.asizeof import asizeof
import numpy as np
from multiprocessing import cpu_count
from sklearn.model_selection import StratifiedShuffleSplit
from helpers.knn_classifier import knn_parameter_search
from helpers.lid_estimators import estimate_intrinsic_dimension
from helpers.dimension_reduction_methods import (
    wrapper_data_projection,
    transform_data_from_model,
    load_dimension_reduction_models
)
from constants import (
    ROOT,
    NEIGHBORHOOD_CONST,
    METRIC_DEF,
    PCA_CUTOFF,
    METHOD_INTRINSIC_DIM,
    METHOD_DIM_REDUCTION
)
try:
    import cPickle as pickle
except:
    import pickle


def extract_layer_embeddings(model, device, train_loader, num_samples=None):
    tot_target = []
    embeddings = []
    num_samples_partial = 0
    # counter = 180   # number of data batches
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        temp = target.detach().cpu().numpy()
        tot_target.extend(temp)
        num_samples_partial += temp.shape[0]
        #print(batch_idx)

        output = model.layer_wise(data)
        if batch_idx > 0:
            for i in range(len(output)):    # each layer
                embeddings[i].append(output[i].detach().cpu().numpy())
        else:
            embeddings = [[v.detach().cpu().numpy()] for v in output]

        if num_samples:
            if num_samples_partial >= num_samples:
                break

    # `embeddings` will be a list of length equal to the number of layers.
    # `embeddings[i]` will be a list of numpy arrays corresponding to the data batches for layer `i`.
    # `embeddings[i][j]` will have shape `(b, d1, d2, d3)` or `(b, d1)` where `b` is the batch size and the rest
    # are dimensions.
    tot_target = np.array(tot_target, dtype=np.int)
    labels_uniq, counts = np.unique(tot_target, return_counts=True)
    for a, b in zip(labels_uniq, counts):
        print("label {}, count = {:d}, proportion = {:.4f}".format(a, b, b / tot_target.shape[0]))

    if (np.max(counts) / np.min(counts)) >= 1.2:
        print("WARNING: classes are not balanced")

    return embeddings, tot_target, counts

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    #parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    #parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    parser.add_argument('--model-type', default='mnist', help='model type')
    #parser.add_argument('--adv-attack', default='FGSM', help='type of adversarial attack')
    #parser.add_argument('--attack', type=bool, default=False, help='launch attack? True or False')
    #parser.add_argument('--distance', type=str, default='inf', help='p norm for attack')
    #parser.add_argument('--train', type=bool, default=False, help='commence training')
    parser.add_argument('--ckpt', type=bool, default=True, help='use ckpt')
    parser.add_argument('--gpu', type=str, default='2', help='gpus to execute code on')
    parser.add_argument('--output', type=str, default='output_layer_extraction.txt', help='output file basename')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    data_path = os.path.join(ROOT, 'data')
    if args.model_type == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_loader = torch.utils.data.DataLoader(datasets.MNIST(data_path, train=True, download=True, transform=transform), batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(datasets.MNIST(data_path, train=False, transform=transform),
                                                  batch_size=args.test_batch_size, shuffle=True, **kwargs)
        model = MNIST().to(device)

    elif args.model_type == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        model = CIFAR10().to(device)
    
    elif args.model_type == 'svhn':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.SVHN(root=data_path, split='train', download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        testset = torchvision.datasets.SVHN(root=data_path, split='test', download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        model = SVHN().to(device)
    
    else:
        print(args.model_type + " not in candidate models; halt!")
        exit()

    if args.ckpt:
        model_path = os.path.join(ROOT, 'models', args.model_type + '_cnn.pt')
        if os.path.exists(model_path):
            if args.model_type == 'mnist':
                model.load_state_dict(torch.load(model_path))
                # embeddings = [[] for _ in range(4)] # 11 layers in the MNIST CNN
            if args.model_type == 'cifar10':
                model.load_state_dict(torch.load(model_path))
                # embeddings = [[] for _ in range(4)] # 11 layers in the MNIST CNN
            if args.model_type == 'svhn':
                model.load_state_dict(torch.load(model_path))
                # embeddings = [[] for _ in range(4)] # 11 layers in the MNIST CNN
        else:
            print(model_path + ' not found')
            exit()
    
        print("empty embeddings list loaded!")

    # Get the feature embeddings from all the layers and the labels
    embeddings, labels, counts = extract_layer_embeddings(model, device, train_loader)
    embeddings_test, labels_test, counts_test = extract_layer_embeddings(model, device, test_loader)

    max_samples = 10000  # number of samples to use for ID estimation and dimension reduction

    # Take a random class-stratified subsample of the data for intrinsic dimension estimation and
    # dimensionality reduction
    sss = StratifiedShuffleSplit(n_splits=1, test_size=max_samples, random_state=args.seed)
    temp = np.zeros((labels.shape[0], 2))   # placeholder data array
    _, indices_sample = next(sss.split(temp, labels))

    #perform some processing on the counts if it is not class balanced
    print("embeddings calculated!")
    n_layers = len(embeddings)
    print("number of layers = {}".format(n_layers))
    #exit()

    # Use half the number of available cores
    cc = cpu_count()
    n_jobs = max(1, int(0.5 * cc))

    output_dir = os.path.join(ROOT, 'outputs', args.model_type)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, args.output)
    lines = []
    # Projection model from the different layers
    model_projection_layers = []
    # Projected (dimension reduced) training and test data from the different layers
    data_train_layers = []
    data_test_layers = []
    for i in range(n_layers):    # number of layers in the CNN
        mode = 'w' if i == 0 else 'a'
        output_fp = open(output_file, mode)
        str0 = "\nLayer: {:d}".format(i + 1)
        print(str0)
        lines.append(str0 + '\n')

        # Embeddings from layer i
        data = np.concatenate(embeddings[i], axis=0)
        s = data.shape
        if len(s) > 2:
            data = data.reshape((s[0], -1))

        data_test = np.concatenate(embeddings_test[i], axis=0)
        s = data_test.shape
        if len(s) > 2:
            data_test = data_test.reshape((s[0], -1))

        if (labels.shape[0] != data.shape[0]) or (labels_test.shape[0] != data_test.shape[0]):
            print("label - sample mismatch; break!")
            exit()
        else:
            print("num labels == num samples; proceeding with intrisic dimensionality calculation!")

        # Random stratified sample from the training portion of the data
        data_sample = data[indices_sample, :]
        labels_sample = labels[indices_sample]

        dim_orig = data_sample.shape[1]
        N_samples = data_sample.shape[0]
        str0 = ("Original dimension = {:d}. Train data size = {:d}. Test data size = {:d}. Sub-sample size used "
                "for dimension reduction = {:d}".format(dim_orig, labels.shape[0], labels_test.shape[0], N_samples))
        print(str0)
        lines.append(str0 + '\n')

        d = estimate_intrinsic_dimension(data_sample, method=METHOD_INTRINSIC_DIM, n_jobs=n_jobs)
        d = min(int(np.ceil(d)), dim_orig)
        str0 = "Intrinsic dimensionality: {:d}".format(d)
        print(str0)
        lines.append(str0 + '\n')

        if d > 20:
            print("\nSearching for the best number of neighbors (k) and projected dimension.")
            d_max = min(10 * d, dim_orig - 1)
            dim_proj_range = np.unique(np.linspace(d, d_max, num=20, dtype=np.int))
            k_max = int(N_samples ** NEIGHBORHOOD_CONST)
            k_range = np.unique(np.linspace(1, k_max, num=10, dtype=np.int))

            k_best, dim_best, error_rate_cv, _, model_projection = knn_parameter_search(
                data_sample, labels_sample, k_range,
                dim_proj_range=dim_proj_range,
                method_proj=METHOD_DIM_REDUCTION,
                metric=METRIC_DEF,
                pca_cutoff=PCA_CUTOFF,
                n_jobs=n_jobs
            )
            str_list = ["k_best: {:d}".format(k_best), "dim_best: {:d}".format(dim_best),
                        "error_rate_cv = {:.6f}".format(error_rate_cv)]
            str0 = '\n'.join(str_list)
            print(str0)
            lines.append(str0 + '\n')
        else:
            print("\nSkipping dimensionality reduction for this layer.")
            model_projection = {'method': METHOD_DIM_REDUCTION, 'mean_data': None, 'transform': None}

        model_projection_layers.append(model_projection)

        # print("\nProjecting the entire train and test data to {:d} dimensions:".format(dim_best))
        # data_train_layers.append(transform_data_from_model(data, model_projection))
        # data_test_layers.append(transform_data_from_model(data_test, model_projection))

        output_fp.writelines(lines)
        output_fp.close()

    print("\nOutputs saved to the file: {}".format(output_file))
    fname = os.path.join(output_dir, 'models_dimension_reduction.pkl')
    with open(fname, 'wb') as fp:
        pickle.dump(model_projection_layers, fp)

    print("Dimension reduction models saved to the file: {}".format(fname))


if __name__ == '__main__':
    main()
