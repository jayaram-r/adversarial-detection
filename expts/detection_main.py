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
    convert_to_loader,
    load_numpy_data,
    get_path_dr_models,
    get_clean_data_path,
    get_adversarial_data_path,
    get_output_path
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
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--model-type', '-m', choices=['mnist', 'cifar10', 'svhn'], default='cifar10',
                        help='model type or name of the dataset')
    parser.add_argument('--seed', '-s', type=int, default=SEED_DEFAULT, help='seed for random number generation')
    parser.add_argument('--detection-method', '--dm', choices=['proposed', 'lid', 'odds', 'dknn'],
                        default='proposed', help='detection method to run')
    parser.add_argument('--test-statistic', '--ts', choices=TEST_STATS_SUPPORTED, default='multinomial',
                        help='type of test statistic to calculate at the layers for the proposed method')
    parser.add_argument('--ood', action='store_true', default=False,
                        help='Perform OOD detection instead of adversarial (if applicable)')
    parser.add_argument('--include-noise', '--in', action='store_true', default=False,
                        help='Include noisy samples in the evaluation')
    parser.add_argument('--model-dim-reduc', '--mdr', default='',
                        help='Path to the saved dimension reduction model file')
    parser.add_argument('--output-dir', '-o', default='', help='directory path for saving the output and model files')
    parser.add_argument('--detection-mechanism', '-dm', default='odds', help='the detection mechanism to use')
    parser.add_argument('--adv-attack', '--aa', choices=['FGSM', 'PGD', 'CW'], default='PGD',
                        help='type of adversarial attack')
    parser.add_argument('--attack-proportion', '--ap', type=float, default=ATTACK_PROPORTION_DEF,
                        help='Proportion of attack samples in the test set (default: {:.2f})'.
                        format(ATTACK_PROPORTION_DEF))
    parser.add_argument('--mixed-attack', '--ma', action='store_true', default=False,
                        help='Use option to enable a mixed attack strategy with multiple methods in '
                             'different proportions')
    parser.add_argument('--num-folds', '--nf', type=int, default=CROSS_VAL_SIZE,
                        help='number of cross-validation folds')
    parser.add_argument('--gpu', type=str, default="2", help='which gpus to execute code on')
    parser.add_argument('--p-norm', '-p', choices=['0', '2', 'inf'], default='inf',
                        help="p norm for the adversarial attack; options are '0', '2' and 'inf'")
    parser.add_argument('--n-jobs', type=int, default=8, help='number of parallel jobs to use for multiprocessing')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if not args.output_dir:
        output_dir = get_output_path(args.model_type)
    else:
        output_dir = args.output_dir

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    apply_dim_reduc = False
    if args.detection_method == 'proposed':
        # Dimension reduction is not applied when the test statistic is 'lid' or 'lle'
        if args.test_statistic == 'multinomial':
            apply_dim_reduc = True

    model_dim_reduc = None
    if apply_dim_reduc:
        if not args.model_dim_reduc:
            # Default path the dimension reduction model file
            model_dim_reduc = get_path_dr_models(args.model_type)

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
        bounds = (-255, 255)

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
        bounds = (-255, 255)

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
        bounds = (-255, 255)

    else:
        raise ValueError("'{}' is not a valid model type".format(args.model_type))

    # Cross-validation folds
    for i in range(args.num_folds):
        # Load the saved clean numpy data from this fold
        numpy_save_path = get_clean_data_path(args.model_type, i + 1)
        data_tr, labels_tr, data_te, labels_te = load_numpy_data(numpy_save_path, adversarial=False)

        # Data loader for the train fold
        train_fold_loader = convert_to_loader(data_tr, labels_tr, batch_size=args.test_batch_size)

        # Data loader for the test fold
        test_fold_loader = convert_to_loader(data_te, labels_te, batch_size=args.test_batch_size)

        # Load the saved adversarial numpy data from this fold
        # TODO: `attack_param_list` needs to be defined
        numpy_save_path = get_adversarial_data_path(args.model_type, i + 1, args.adv_attack, attack_param_list)
        data_tr_adv, labels_tr_adv, data_te_adv, labels_te_adv = load_numpy_data(numpy_save_path, adversarial=True)

        # Adversarial data loader for the train fold
        adv_train_fold_loader = convert_to_loader(data_tr_adv, labels_tr_adv, batch_size=args.test_batch_size)

        # Adversarial data loader for the test fold
        adv_test_fold_loader = convert_to_loader(data_te_adv, labels_te_adv, batch_size=args.test_batch_size)


        if args.detection_mechanism == 'odds':
            # call functions from detectors/detector_odds_are_odd.py
            train_inputs = (data_tr, labels_tr)
            predictor = fit_odds_are_odd(train_inputs, model, args.model_type, num_classes, with_attack=True)
            next(predictor)
            detect_odds_are_odd(predictor, test_fold_loader, adv_loader, model)

        elif args.detection_mechanism == 'lid':
            # to do: jayaram will complete the procedure
            # required methods are in `detectors.detector_lid_paper`
            continue
        elif args.detection_mechanism == 'proposed':
            # to do: jayaram will complete the procedure
            # required methods are in `detectors.detector_proposed`
            continue
        elif args.detection_mechanism == 'dknn':
            continue


if __name__ == '__main__':
    main()
