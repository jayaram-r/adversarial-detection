"""
https://github.com/yk/icml19_public
"""
from nets.svhn import *
from nets.cifar10 import *
from nets.mnist import *
from nets.resnet import *
import torch
import torchvision


def get_samples_as_ndarray(loader):
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.cpu().numpy(), target.cpu().numpy()
        target = target.reshape((target.shape[0], 1))
        if batch_idx == 0:
            X, Y = data, target
        else:
            X = np.vstack((X,data))
            Y = np.vstack((Y,target))

    Y = Y.reshape((-1,))
    return X,Y


def latent_and_logits_fn(x):
    lat, log = net_forward(x, True)[-2:]
    lat = lat.reshape(lat.shape[0], -1)
    return lat, log


def get_wcls(model_type):
    if model_type == 'mnist':
        w_cls = model.fc2.weight
    elif model_type == 'cifar10':
        w_cls = list(model.children())[-1].weight
    elif model_type == 'svhn':
        w_cls = model.fc3.weight
    else:
        raise ValueError("Invalid model type '{}'".format(model_type))

    return w_cls


def return_data(test_loader):
    X, Y = get_samples_as_ndarray(test_loader)
    #pgd_train = foolbox_attack(model, device, test_loader, bounds, p_norm=args.p_norm, adv_attack='PGD') #verify if all parameters are defined
    pgd_train = None #to mirror the ICML'19 repo
    return X, Y, pgd_train
