"""
Main script for running the adversarial and OOD detection experiments.
"""
from __future__ import absolute_import, division, print_function
import sys
import argparse
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold
from nets.mnist import *
from nets.cifar10 import *
from nets.svhn import *
from nets.resnet import *
from helpers.constants import (
    ROOT,
    SEED_DEFAULT,
    CROSS_VAL_SIZE,
    ATTACK_PROPORTION_DEF,
    NORMALIZE_IMAGES,
    TEST_STATS_SUPPORTED
)
from detectors.tf_robustify import collect_statistics
from helpers.utils import (
    load_model_checkpoint,
    save_model_checkpoint,
<<<<<<< HEAD
    convert_to_list,
    convert_to_loader,
    verify_data_loader
=======
    convert_to_loader
>>>>>>> upstream/master
)
from helpers.attacks import foolbox_attack, foolbox_attack_helper
from detectors.detector_odds_are_odd import (
    get_samples_as_ndarray,
    get_wcls,
    return_data,
    fit_odds_are_odd,
    detect_odds_are_odd
)
from detectors.detector_lid_paper import (
    flip,
    get_noisy_samples,
    DetectorLID
)   # ICLR 2018
from detectors.detector_proposed import DetectorLayerStatistics, extract_layer_embeddings
from detectors.detector_deep_knn import DeepKNN


# Proportion of attack samples from each method when a mixed attack strategy is used at test time.
# The proportions should sum to 1. Note that this is a proportion of the subset of attack samples and not a
# proportion of all the test samples.
MIXED_ATTACK_PROPORTIONS = {
    'FGSM': 0.1,
    'PGD': 0.4,
    'CW': 0.5
}


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-batch-size', '--tb', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--output-dir', '-o', default='', help='directory path for saving the output and model files')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--model-type', '-m', choices=['mnist', 'cifar10', 'svhn'], default='cifar10',
                        help='model type or name of the dataset')
    parser.add_argument('--seed', '-s', type=int, default=SEED_DEFAULT, help='seed for random number generation')
    parser.add_argument('--ckpt', default=False, help='to use checkpoint or not')
    parser.add_argument('--adv-attack', '--aa', choices=['FGSM', 'PGD', 'CW'], default='PGD',
                        help='type of adversarial attack')
    parser.add_argument('--gpu', type=str, default="2", help='which gpus to execute code on')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size of evaluation')
    parser.add_argument('--p-norm', '-p', choices=['0', '2', 'inf'], default='inf',
                        help="p norm for the adversarial attack; options are '0', '2' and 'inf'")
    parser.add_argument('--stepsize', type=float, default=0.001, help='stepsize')
    parser.add_argument('--confidence', type=float, default=0, help='confidence needed by CW')
    parser.add_argument('--epsilon', type=float, default=0.3, help='epsilon value')
    parser.add_argument('--max-iterations', type=float, default=1000, help='max num. of iterations')
    parser.add_argument('--iterations', type=float, default=40, help='num. of iterations')
    parser.add_argument('--max-epsilon', type=float, default=1, help='max. value of epsilon')
    parser.add_argument('--num-folds', '--nf', type=int, default=CROSS_VAL_SIZE,
                        help='number of cross-validation folds')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
   
    bounds_master = (-255,255)

    if not args.output_dir:
        output_dir = os.path.join(ROOT, 'numpy_data', args.model_type)
    else:
        output_dir = args.output_dir

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    data_path = os.path.join(ROOT, 'data')

    if args.model_type == 'mnist':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(*NORMALIZE_IMAGES['mnist'])]
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train=False, download=True, transform=transform),
            batch_size=args.test_batch_size, shuffle=True, **kwargs
        )
        model = MNIST().to(device)
        model = load_model_checkpoint(model, args.model_type)
        num_classes = 10
        bounds=bounds_master

    elif args.model_type == 'cifar10':
        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(*NORMALIZE_IMAGES['cifar10'])]
        )
        testset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        num_classes = 10
        model = ResNet34().to(device)
        model = load_model_checkpoint(model, args.model_type)
        bounds=bounds_master

    elif args.model_type == 'svhn':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(*NORMALIZE_IMAGES['svhn'])]
        )
        testset = datasets.SVHN(root=data_path, split='test', download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        num_classes = 10
        model = SVHN().to(device)
        model = load_model_checkpoint(model, args.model_type)
        bounds=bounds_master

    else:
        raise ValueError("'{}' is not a valid model type".format(args.model_type))

    #convert the data loader to 2 ndarrays
    data, labels = get_samples_as_ndarray(test_loader)
   
    #obtain batch_size
    batch_size = args.batch_size

    #verify if the data loader is the same as the ndarrays it generates
    print(verify_data_loader(test_loader, batch_size = batch_size)) #(True, True)
    
    # Stratified cross-validation split
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
   
    #fold number
    i = 1

    #repeat for each fold in the split
    for ind_tr, ind_te in skf.split(data, labels): 
        data_tr = data[ind_tr, :]
        labels_tr = labels[ind_tr]
        data_te = data[ind_te, :]
        labels_te = labels[ind_te]
        
        #make dir based on fold to save data
        numpy_save_path = os.path.join(output_dir, "fold_"+str(i))
        if os.path.isdir(numpy_save_path) == False:
            os.makedirs(numpy_save_path)

        adv_save_path = os.path.join(os.path.join(output_dir, args.adv_attack), "fold_"+str(i))
        if os.path.isdir(adv_save_path) == False:
            os.makedirs(adv_save_path) 
        
        #save train fold
        np.save(os.path.join(numpy_save_path, 'data_tr.npy'), data_tr)
        np.save(os.path.join(numpy_save_path, 'labels_tr.npy'), labels_tr)
        
        #save test fold
        np.save(os.path.join(numpy_save_path, 'data_te.npy'), data_te)
        np.save(os.path.join(numpy_save_path, 'labels_te.npy'), labels_te)

        #print prompt
        print("saved train and test fold for fold:", i)
        data_tr_list, labels_tr_list = convert_to_list(data_tr), convert_to_list(labels_tr)

        #convert ndarray to list
        data_te_list, labels_te_list = convert_to_list(data_te), convert_to_list(labels_te)

        #convert list to loader
        #test_loader = convert_to_loader(data_te, labels_te, batch_size=args.test_batch_size)
        test_loader = convert_to_loader(data_te_list, labels_te_list)
        #train_loader = convert_to_loader(data_te, labels_te, batch_size=args.test_batch_size)
        train_loader = convert_to_loader(data_tr_list, labels_tr_list)

            
        #setting adv. attack parameters
        stepsize=args.stepsize
        confidence=args.confidence
        epsilon=args.epsilon
        max_iterations=args.max_iterations
        iterations=args.iterations
        max_epsilon=args.max_epsilon
        
        #convert list to loader
        test_loader = convert_to_loader(data_te, labels_te, batch_size=args.test_batch_size)

        #use dataloader to create adv. examples; adv_inputs is an ndarray
        adv_inputs, adv_labels = foolbox_attack(model, 
                device, 
                test_loader, 
                bounds, 
                num_classes=num_classes, 
                p_norm=args.p_norm, 
                adv_attack=args.adv_attack, 
                stepsize=stepsize,
                confidence=confidence,
                epsilon=epsilon,
                max_iterations=max_iterations,
                iterations=iterations,
                max_epsilon=max_epsilon,
                labels_req=True)
        
        #save test fold's adv. examples
        np.save(os.path.join(adv_save_path, 'data_te_adv.npy'), adv_inputs)
        np.save(os.path.join(adv_save_path, 'labels_te_adv.npy'), adv_labels)

        print("saved adv. examples generated by test fold for fold:",i)
        exit()

        adv_inputs, adv_labels = foolbox_attack(model, 
                device, 
                train_loader, 
                bounds, 
                num_classes=num_classes, 
                p_norm=args.p_norm, 
                adv_attack=args.adv_attack,
                stepsize=stepsize,
                confidence=confidence,
                epsilon=epsilon,
                max_iterations=max_iterations,
                iterations=iterations,
                max_epsilon=max_epsilon,
                adv_attack=args.adv_attack, 
                labels_req=True)
        
        #save train_fold's adv. examples
        np.save(os.path.join(adv_save_path, 'data_tr_adv.npy'), adv_inputs)
        np.save(os.path.join(adv_save_path, 'labels_tr_adv.npy'), adv_labels)

        print("saved adv. examples generated by train fold for fold:",i)

        i = i + 1

if __name__ == '__main__':
    main()
