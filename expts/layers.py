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
from helpers.knn_classifier import *
from helpers.lid_estimators import *

def extract(args, model, device, train_loader):
    tot_target = []
    embeddings = []
    counter = 180
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx < counter:
            data, target = data.to(device), target.to(device)
            target_numpy = target.detach().cpu().numpy().tolist()
            tot_target = tot_target + target_numpy
            #print(batch_idx)

            output = model.layer_wise(data)
            if batch_idx > 0:
                for i in range(len(output)):    # each layer
                    embeddings[i].append(output[i].detach().cpu().numpy())
            else:
                embeddings = [[v.detach().cpu().numpy()] for v in output]

    counts = [0] * 10
    for i in range(10):
        counts[i] = tot_target.count(i)
        print("label: ",i,"occurence = ", counts[i])

    if (max(counts) - min(counts)) > 200:
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
    data_path = ROOT+'/data' 
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
        model_path = ROOT + '/models/' + args.model_type + '_cnn.pt'
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
            print(model_path+' not found')
            exit()
    
        print("empty embeddings list loaded!")

    embeddings, labels, counts = extract(args, model, device, train_loader)
    #perform some processing on the counts if it is not class balanced
    print("embeddings calculated!")
    n_layers = len(embeddings)
    print("number of layers = {}".format(n_layers))
    #exit()

    output_fp = open(ROOT + '/output/' + args.output, "a")
    for i in range(n_layers): #will be equal to number of layers in the CNN
        print("begin processing for layer:", i)
        data = None
        for j in range(len(embeddings[i])): #will be equal to len(train_loader) 
            embedding_shape = embeddings[i][j].shape
            num_samples = embedding_shape[0]
            
            if len(embedding_shape) > 2:
                temp = embeddings[i][j].reshape((num_samples, -1))
            else:
                temp = embeddings[i][j]

            if data is None:
                data = temp
            else:
                data = np.vstack((data, temp))

        str0 = "\nLayer: " + str(i+1)
        print(str0)
        #jayaram's functions
        #_ = estimate_intrinsic_dimension(temp, method='two_nn', n_jobs=16)
        labels = np.asarray(labels)
        N_samples = data.shape[0]
        if labels.shape[0] != N_samples:
            print("label - sample mismatch; break!")
            exit()
        else:
            print("num labels == num samples; proceeding with intrisic dimensionality calculation!")

        d = estimate_intrinsic_dimension(data, method='lid_mle', n_jobs=16)
        d = int(np.ceil(d))
        str1 = "intrinsic dimensionality: " + str(d)
        print(str1)
        metric = 'cosine'
        pca_cutoff = 0.995
        method_proj = 'NPP'
        n_jobs = 16
        dim_proj_range = np.linspace(d, 10 * d, num=20, dtype=np.int)
        k_max = int(N_samples ** 0.4)
        k_range = np.linspace(1, k_max, num=10, dtype=np.int)
        
        k_best, dim_best, error_rate_cv, data_proj = knn_parameter_search(
            data, labels, k_range,
            dim_proj_range = dim_proj_range,
            method_proj=method_proj,
            metric = metric,
            pca_cutoff = pca_cutoff,
            n_jobs = n_jobs
        )
        str2 = "k_best: " + str(k_best) + " dim_best: " + str(dim_best) + " error_rate_cv: " + str(error_rate_cv)
        #print("error_rate_cv:", error_rate_cv)
        #print("data_proj:", data_proj)
        #print(str1)
        print(str2)
        output_fp.write(str0 + "\n")
        output_fp.write(str1 + "\n")
        output_fp.write(str2 + "\n")

    output_fp.close()

if __name__ == '__main__':
    main()
