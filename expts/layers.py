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
'''
from helpers.knn_classifier import knn_parameter_search
from helpers.lid_estimators import estimate_intrinsic_dimension
from helpers.constants import (
    NEIGHBORHOOD_CONST,
    METRIC_DEF,
    PCA_CUTOFF,
    METHOD_INTRINSIC_DIM,
    METHOD_DIM_REDUCTION
)
'''

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
    parser.add_argument('--output', type=str, default='mnist.txt', help='output')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    ROOT = '/nobackup/varun/adversarial-detection/expts'
    data_path = os.path.join(ROOT, 'data')
    if args.model_type == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ])
        train_loader = torch.utils.data.DataLoader(datasets.MNIST(data_path, train=True, download=True, transform=transform), batch_size=args.batch_size, shuffle=True, **kwargs)
        model = MNIST().to(device)

    elif args.model_type == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        model = CIFAR10().to(device)
    
    elif args.model_type == 'svhn':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.SVHN(root=data_path, split='train', download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        model = SVHN().to(device)
    
    else:
        print(args.model_type+" not in candidate models; halt!")
        exit()

    if args.ckpt:
        model_path = os.path.join(ROOT, 'models', args.model_type, '_cnn.pt')
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

    max_samples = 10000     # number of samples to use for ID estimation and dimension reduction
    embeddings, labels, counts = extract_layer_embeddings(model, device, train_loader, num_samples=max_samples)
    #perform some processing on the counts if it is not class balanced
    print("embeddings calculated!")
    n_layers = len(embeddings)
    print("number of layers = {}".format(n_layers))
    exit()

    # Use half the number of available cores
    cc = cpu_count()
    n_jobs = max(1, int(0.5 * cc))

    output_dir = os.path.join(ROOT, 'output')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, args.output)
    output_fp = open(output_file, "w")
    lines = []
    for i in range(n_layers): #will be equal to number of layers in the CNN
        str0 = "\nLayer: {}".format(i + 1)
        print(str0)
        lines.append(str0 + '\n')

        # Embeddings from layer i
        data = np.concatenate(embeddings[i], axis=0)
        s = data.shape
        if len(s) > 2:
            data = data.reshape((s[0], -1))

        #jayaram's functions
        N_samples = data.shape[0]
        str0 = "Number of samples: {:d}".format(N_samples)
        print(str0)
        lines.append(str0 + '\n')
        if labels.shape[0] != N_samples:
            print("label - sample mismatch; break!")
            exit()
        else:
            print("num labels == num samples; proceeding with intrisic dimensionality calculation!")

        d = estimate_intrinsic_dimension(data, method=METHOD_INTRINSIC_DIM, n_jobs=n_jobs)
        d = int(np.ceil(d))
        str0 = "Intrinsic dimensionality: {:d}".format(d)
        print(str0)
        lines.append(str0 + '\n')

        print("\nSearching for the best number of neighbors (k) and projected dimension.")
        d_max = min(10 * d, data.shape[1] - 1)
        dim_proj_range = np.linspace(d, d_max, num=20, dtype=np.int)
        k_max = int(N_samples ** NEIGHBORHOOD_CONST)
        k_range = np.linspace(1, k_max, num=10, dtype=np.int)

        k_best, dim_best, error_rate_cv, data_proj = knn_parameter_search(
            data, labels, k_range,
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

    output_fp.writelines(lines)
    print("Outputs saved to the file: {}".format(output_file))
    output_fp.close()


if __name__ == '__main__':
    main()
