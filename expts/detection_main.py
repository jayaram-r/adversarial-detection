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
    convert_to_list,
    convert_to_loader
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
    parser.add_argument('--ckpt', default=False, help='to use checkpoint or not')
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
    parser.add_argument('--p-norm', '-p', choices=['2', 'inf', '0'], default='inf',
                        help="p norm for the adversarial attack; options are '2', 'inf', and '0'")
    parser.add_argument('--n-jobs', type=int, default=8, help='number of parallel jobs to use for multiprocessing')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if not args.output_dir:
        output_dir = os.path.join(ROOT, 'outputs', args.model_type)
    else:
        output_dir = args.output_dir

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    apply_dim_reduc = False
    if args.detection_method == 'proposed':
        # Dimension reduction is not applied when the test statistic is 'lid' or 'lpp'
        if args.test_statistic == 'multinomial':
            apply_dim_reduc = True

    model_dim_reduc = None
    if apply_dim_reduc:
        if not args.model_dim_reduc:
            model_dim_reduc = os.path.join(ROOT, 'outputs', args.model_type, 'models_dimension_reduction.pkl')

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

    # Only the test split of the data will be used for detection experiments
    # convert the data loader to 2 ndarrays
    data_clean, labels_clean = get_samples_as_ndarray(test_loader)
    
    # Stratified cross-validation split
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

    scores_adver_all = np.zeros(labels_clean.shape[0])
    #repeat for each fold in the split
    for j, (ind_tr, ind_te) in enumerate(skf.split(data_clean, labels_clean)):
        data_tr_init = data_clean[ind_tr, :]
        labels_tr_init = labels_clean[ind_tr]
        data_te_init = data_clean[ind_te, :]
        labels_te_init = labels_clean[ind_te]
        print("\nCross-validation fold {:d}. Train split size = {:d}. Test split size = {:d}".
              format(j + 1, labels_tr_init.shape[0], labels_te_init.shape[0]))

        # Data loader for the train data split
        train_fold_loader = convert_to_loader(data_tr_init, labels_tr_init, batch_size=args.test_batch_size)

        print("\nCalculating the layer embeddings and DNN predictions for the train data split:")
        layer_embeddings_tr, labels_tr, labels_pred_tr, _ = extract_layer_embeddings(
            model, device, train_fold_loader, method=args.detection_mechanism
        )

        # Data loader for the test data split
        # data_te_list = convert_to_list(data_te_init)
        # labels_te_list = convert_to_list(labels_te_init)
        test_fold_loader = convert_to_loader(data_te_init, labels_te_init, batch_size=args.test_batch_size)

        print("\nCalculating the layer embeddings and DNN predictions for the test data split:")
        layer_embeddings_te, labels_te, labels_pred_te, _ = extract_layer_embeddings(
            model, device, test_fold_loader, method=args.detection_mechanism
        )

        # TODO: generate adversarial samples from the training split because this is used by the LID detection method
        # TODO: generate noisy samples from the training split because this is used by the LID detection method

        # use the test data loader to create adv. examples from the test split; adv_inputs is an ndarray
        adv_inputs, adv_labels = foolbox_attack(model, device, test_fold_loader, bounds, num_classes=num_classes,
                                                p_norm=args.p_norm, adv_attack=args.adv_attack, labels_req=True)

        # adv_inputs_list, adv_labels_list = convert_to_list(adv_inputs), convert_to_list(adv_labels)
        # convert adversarial array inputs to torch dataloader
        adv_loader = convert_to_loader(adv_inputs, adv_labels, batch_size=args.test_batch_size)
 
        if args.detection_mechanism == 'odds':
            # call functions from detectors/detector_odds_are_odd.py
            train_inputs = (data_tr_init, labels_tr_init)
            predictor = fit_odds_are_odd(train_inputs, model, args.model_type, num_classes, with_attack=True)
            next(predictor)
            detect_odds_are_odd(predictor, test_fold_loader, adv_loader, model)

        elif args.detection_mechanism == 'lid':
            # TODO: generate adversarial and noisy data from the clean training fold
            model_det = DetectorLID(
                n_neighbors=None,
                n_jobs=args.n_jobs,
                seed_rng=args.seed
            )
            model_det, _, _, _ = model_det.fit(layer_embeddings_tr, layer_embeddings_noisy_tr,
                                               layer_embeddings_adversarial_tr)
            scores_adver = model_det.score(layer_embeddings_te)

        elif args.detection_mechanism == 'proposed':
            det_model = DetectorLayerStatistics(
                layer_statistic=args.test_statistic,
                skip_dim_reduction=(not apply_dim_reduc),
                model_file_dim_reduction=model_dim_reduc,
                n_jobs=args.n_jobs,
                seed_rng=args.seed
            )
            det_model = det_model.fit(layer_embeddings_tr, labels_tr, labels_pred_tr)
            scores_adver, scores_ood = det_model.score(layer_embeddings_te, labels_pred_te)

        elif args.detection_mechanism == 'dknn':
            det_model = DeepKNN(
                n_neighbors=None,   # can be set to other values used in the paper
                skip_dim_reduction=True,
                model_file_dim_reduction=model_dim_reduc,
                n_jobs=args.n_jobs,
                seed_rng=args.seed
            )
            det_model = det_model.fit(layer_embeddings_tr, labels_tr)
            scores_adver, labels_pred_dknn = det_model.score(layer_embeddings_te)


if __name__ == '__main__':
    main()
