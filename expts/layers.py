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

def extract(args, model, device, train_loader, layers):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model.layer_wise(data)
        for i in range(len(layers)):
            layers[i].append(output[i].detach().cpu().numpy())
    return layers

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
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    ROOT = '/nobackup/varun/adversarial-detection/expts' 
    
    if args.model_type == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ])
        train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=args.batch_size, shuffle=True, **kwargs)
        model = MNIST().to(device)
    
    elif args.model_type == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        model = CIFAR10().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    elif args.model_type == 'svhn':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        model = SVHN().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    
    else:
        print(args.model_type+" not in candidate models; halt!")
        exit()

    if args.ckpt:
        model_path = ROOT + '/models/'+args.model_type+'_cnn.pt'
        if os.path.exists(model_path) == True:
            if args.model_type == 'mnist':
                model.load_state_dict(torch.load(model_path))
                layers = [[] for i in range(11)]
            if args.model_type == 'cifar10':
                model.load_state_dict(torch.load(model_path))
            if args.model_type == 'svhn':
                model.load_state_dict(torch.load(model_path))
        else:
            print(model_path+' not found')
            exit()
    
    layers = extract(args, model, device, train_loader, layers)
    for i in range(len(layers)):
        for j in range(len(layers[i])):
            if j == 0:
                print(layers[i][j].shape)


if __name__ == '__main__':
    main()
