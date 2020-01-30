"""
Main script for running the adversarial and OOD detection experiments.
"""
from __future__ import absolute_import, division, print_function
import sys
import argparse
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from helpers.constants import (
    ROOT,
    SEED_DEFAULT,
    CROSS_VAL_SIZE,
    ATTACK_PROPORTION_DEF
)
from helpers.tf_robustify import *


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
    parser.add_argument('--model-type', '-m', choices=['mnist', 'cifar10', 'svhn'], default='mnist',
                        help='model type or name of the dataset')
    parser.add_argument('--seed', '-s', type=int, default=SEED_DEFAULT, help='seed for random number generation')
    parser.add_argument('--detection-method', '--dm', choices=['proposed', 'lid', 'odds_are_odd'],
                        default='proposed', help='detection method to run')
    parser.add_argument('--test-statistic', '--ts', choices=['multinomial', 'lid'], default='multinomial',
                        help='type of test statistic to calculate at the layers for the proposed method')
    parser.add_argument('--ood', action='store_true', default=False,
                        help='Perform OOD detection instead of adversarial (if applicable)')
    parser.add_argument('--include-noise', '--in', action='store_true', default=False,
                        help='Include noisy samples in the evaluation')
    parser.add_argument('--model-dim-reduc', '--mdr', default='',
                        help='Path to the saved dimension reduction model file')
    parser.add_argument('--output-dir', '-o', default='', help='directory path for saving the output and model files')
    parser.add_argument('--detection-mechanism', '-dm', default='odds', help='the detection mechanism to use')
    parser.add_argument('--ckpt', default=False, help='to use checkpoint or not')
    parser.add_argument('--adv-attack', '--aa', choices=['FGSM', 'PGD', 'CW'], default='FGSM',
                        help='type of adversarial attack')
    parser.add_argument('--attack-proportion', '--ap', type=float, default=ATTACK_PROPORTION_DEF,
                        help='Proportion of attack samples in the test set (default: {:.2f})'.
                        format(ATTACK_PROPORTION_DEF))
    parser.add_argument('--mixed-attack', '--ma', action='store_true', default=False,
                        help='Use option to enable a mixed attack strategy with multiple methods in '
                             'different proportions')
    parser.add_argument('--num-folds', '--nf', type=int, default=CROSS_VAL_SIZE,
                        help='number of cross-validation folds')
    args = parser.parse_args()

    if not args.output_dir:
        output_dir = os.path.join(ROOT, 'outputs', args.model_type)
    else:
        output_dir = args.output_dir

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    apply_dim_reduc = False
    if args.detection_method == 'proposed':
        # Dimension reduction is not applied when the test statistic is LID
        if args.test_statistic == 'multinomial':
            apply_dim_reduc = True

    if apply_dim_reduc:
        if not args.model_dim_reduc:
            model_dim_reduc = os.path.join(ROOT, 'outputs', args.model_type, 'models_dimension_reduction.pkl')


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

    else:
        raise ValueError("'{}' is not a valid model type".format(args.model_type))



    """
    # Stratified cross-validation split
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    for ind_tr, ind_te in skf.split(data, labels):
        data_tr = data[ind_tr, :]
        labels_tr = labels[ind_tr]
        data_te = data[ind_te, :]
        labels_te = labels[ind_te]
    """


    if args.detection_mechanism == 'odds':
        # to do: what are the X,Y values needed?
        predictor = collect_statistics(X, Y, latent_and_logits_fn_th=latent_and_logits_fn, nb_classes=num_classes,
                weights=w_cls, cuda=args.cuda, debug=args.debug, targeted=args.collect_targeted, noise_eps=args.noise_eps.split(','),
                noise_eps_detect=noise_eps_detect.split(','), num_noise_samples=args.wdiff_samples, batch_size=args.eval_bs,
                pgd_eps=args.eps, pgd_lr=args.attack_lr, pgd_iters=args.iters, clip_min=clip_min, clip_max=clip_max,
                p_ratio_cutoff=args.maxp_cutoff, save_alignments_dir='logs/stats' if args.save_alignments else None,
                load_alignments_dir=os.path.expanduser(ROOT+'/data/advhyp/{}/stats'.format(args.model)) if args.load_alignments else None,
                clip_alignments=args.clip_alignments, pgd_train=pgd_train, fit_classifier=args.fit_classifier, just_detect=args.just_detect)


if __name__ == '__main__':
    main()
