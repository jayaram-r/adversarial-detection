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
# from nets.cifar10 import *
from nets.svhn import *
from nets.resnet import *
from helpers.constants import (
    ROOT,
    SEED_DEFAULT,
    CROSS_VAL_SIZE,
    ATTACK_PROPORTION_DEF,
    NORMALIZE_IMAGES,
    TEST_STATS_SUPPORTED,
    FPR_MAX_PAUC,
    FPR_THRESH
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
    get_output_path,
    metrics_detection,
    metrics_varying_positive_class_proportion
)
from helpers.attacks import foolbox_attack
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


ATTACK_PARAMS = {
    'stepsize': 0.001,
    'confidence': 0,
    'epsilon': 0.003,
    'maxiterations': 1000,
    'iterations': 40,
    'maxepsilon': 1
}

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
        # bounds = (-255, 255)

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
        # bounds = (-255, 255)

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
        # bounds = (-255, 255)

    else:
        raise ValueError("'{}' is not a valid model type".format(args.model_type))

    attack_params_list = [
        ('stepsize', ATTACK_PARAMS['stepsize']),
        ('confidence', ATTACK_PARAMS['confidence']),
        ('epsilon', ATTACK_PARAMS['epsilon']),
        ('maxiterations', ATTACK_PARAMS['maxiterations']),
        ('iterations', ATTACK_PARAMS['iterations']),
        ('maxepsilon', ATTACK_PARAMS['maxepsilon']),
        ('pnorm', args.p_norm)
    ]

    # Cross-validation
    auc_scores = np.zeros(args.num_folds)
    avg_precision_scores = np.zeros(args.num_folds)
    pauc_scores = np.zeros((args.num_folds, len(FPR_MAX_PAUC)))
    tpr_scores = np.zeros((args.num_folds, len(FPR_THRESH)))
    fpr_scores = np.zeros_like(tpr_scores)
    for i in range(args.num_folds):
        # Load the saved clean numpy data from this fold
        numpy_save_path = get_clean_data_path(args.model_type, i + 1)
        data_tr, labels_tr, data_te, labels_te = load_numpy_data(numpy_save_path, adversarial=False)

        print("\nCross-validation fold {:d}. Train split size = {:d}. Test split size = {:d}".
              format(i + 1, labels_tr.shape[0], labels_te.shape[0]))
        # Data loader for the train fold
        train_fold_loader = convert_to_loader(data_tr, labels_tr, batch_size=args.test_batch_size)

        # Data loader for the test fold
        test_fold_loader = convert_to_loader(data_te, labels_te, batch_size=args.test_batch_size)

        print("\nCalculating the layer embeddings and DNN predictions for the clean train data split:")
        layer_embeddings_tr, labels_tr1, labels_pred_tr, _ = extract_layer_embeddings(
            model, device, train_fold_loader, method=args.detection_method
        )
        # NOTE: `labels_tr1` returned by this function should be the same as the `labels_tr` loaded from file
        # because the DataLoader has `shuffle` set to False.
        if not np.array_equal(labels_tr, labels_tr1):
            raise ValueError("Class labels returned by 'extract_layer_embeddings' is different from the original "
                             "labels.")

        print("\nCalculating the layer embeddings and DNN predictions for the clean test data split:")
        layer_embeddings_te, labels_te1, labels_pred_te, _ = extract_layer_embeddings(
            model, device, test_fold_loader, method=args.detection_method
        )
        if not np.array_equal(labels_te, labels_te1):
            raise ValueError("Class labels returned by 'extract_layer_embeddings' is different from the original "
                             "labels.")

        # Load the saved adversarial numpy data generated from this training and test fold
        numpy_save_path = get_adversarial_data_path(args.model_type, i + 1, args.adv_attack, attack_params_list)
        data_tr_adv, labels_tr_adv, data_te_adv, labels_te_adv = load_numpy_data(numpy_save_path, adversarial=True)

        num_adv_tr = labels_tr_adv.shape[0]
        num_adv_te = labels_te_adv.shape[0]
        print("\nNumber of adversarial samples generated from the train fold = {:d}.".format(num_adv_tr))
        print("Number of adversarial samples generated from the test fold = {:d}.".format(num_adv_te))
        print("Percentage of adversarial samples in the test fold = {:.4f}.".
              format((100. * num_adv_te) / labels_te.shape[0]))

        # Adversarial data loader for the train fold
        adv_train_fold_loader = convert_to_loader(data_tr_adv, labels_tr_adv, batch_size=args.test_batch_size)

        # Adversarial data loader for the test fold
        adv_test_fold_loader = convert_to_loader(data_te_adv, labels_te_adv, batch_size=args.test_batch_size)

        if args.detection_method == 'lid':
            # Needed only for the LID method
            print("\nCalculating the layer embeddings and DNN predictions for the adversarial train data split:")
            layer_embeddings_tr_adv, labels_tr_adv1, labels_pred_tr_adv, _ = extract_layer_embeddings(
                model, device, adv_train_fold_loader, method=args.detection_method
            )
            if not np.array_equal(labels_tr_adv, labels_tr_adv1):
                raise ValueError("Class labels returned by 'extract_layer_embeddings' is different from the "
                                 "original labels.")

        print("\nCalculating the layer embeddings and DNN predictions for the adversarial test data split:")
        layer_embeddings_te_adv, labels_te_adv1, labels_pred_te_adv, _ = extract_layer_embeddings(
            model, device, adv_test_fold_loader, method=args.detection_method
        )
        if not np.array_equal(labels_te_adv, labels_te_adv1):
            raise ValueError("Class labels returned by 'extract_layer_embeddings' is different from the original "
                             "labels.")

        # Detection labels (0 denoting clean and 1 adversarial)
        labels_detec = np.concatenate([np.zeros(labels_pred_te.shape[0], dtype=np.int),
                                       np.ones(labels_pred_te_adv.shape[0], dtype=np.int)])
        if args.detection_method == 'odds':
            # call functions from detectors/detector_odds_are_odd.py
            train_inputs = (data_tr, labels_tr)
            predictor = fit_odds_are_odd(train_inputs, model, args.model_type, num_classes, with_attack=True)
            next(predictor)
            detections_clean, detections_attack = detect_odds_are_odd(predictor, test_fold_loader,
                                                                      adv_test_fold_loader, model)
            scores_adv = np.concatenate([detections_clean, detections_attack])
            # Unlike the other methods, these are not continuous valued scores

        elif args.detection_method == 'lid':
            # TODO: generate noisy data from the clean training fold data. Setting this to `None` will inform
            # the detector to skip noisy data
            layer_embeddings_tr_noisy = None

            model_det = DetectorLID(
                n_neighbors=None,
                n_jobs=args.n_jobs,
                seed_rng=args.seed
            )
            # Fit the detector on clean, noisy, and adversarial data from the training fold
            ret = model_det.fit(layer_embeddings_tr, layer_embeddings_tr_adv,
                                layer_embeddings_noisy=layer_embeddings_tr_noisy)
            # Scores on clean data from the test fold
            scores_adv1 = model_det.score(layer_embeddings_te)

            # Scores on adversarial data from the test fold
            scores_adv2 = model_det.score(layer_embeddings_te_adv)

            scores_adv = np.concatenate([scores_adv1, scores_adv2])

        elif args.detection_method == 'proposed':
            det_model = DetectorLayerStatistics(
                layer_statistic=args.test_statistic,
                skip_dim_reduction=(not apply_dim_reduc),
                model_file_dim_reduction=model_dim_reduc,
                n_jobs=args.n_jobs,
                seed_rng=args.seed
            )
            # Fit the detector on clean data from the training fold
            det_model = det_model.fit(layer_embeddings_tr, labels_tr, labels_pred_tr)

            # Scores on clean data from the test fold
            scores_adv1, scores_ood1 = det_model.score(layer_embeddings_te, labels_pred_te)

            # Scores on adversarial data from the test fold
            scores_adv2, scores_ood2 = det_model.score(layer_embeddings_te_adv, labels_pred_te_adv)

            scores_adv = np.concatenate([scores_adv1, scores_adv2])
            scores_ood = np.concatenate([scores_ood1, scores_ood2])

        elif args.detection_method == 'dknn':
            det_model = DeepKNN(
                n_neighbors=None,  # can be set to other values used in the paper
                skip_dim_reduction=True,
                model_file_dim_reduction=model_dim_reduc,
                n_jobs=args.n_jobs,
                seed_rng=args.seed
            )
            # Fit the detector on clean data from the training fold
            det_model = det_model.fit(layer_embeddings_tr, labels_tr)

            # Scores on clean data from the test fold
            scores_adv1, labels_pred_dknn1 = det_model.score(layer_embeddings_te)

            # Scores on adversarial data from the test fold
            scores_adv2, labels_pred_dknn2 = det_model.score(layer_embeddings_te_adv)

            scores_adv = np.concatenate([scores_adv1, scores_adv2])

        # Performance metrics
        print("\nDetection performance metrics on fold {:d}:".format(i + 1))
        auc_scores[i], pauc_scores[i, :], avg_precision_scores[i], tpr_scores[i, :], fpr_scores[i, :] = \
            metrics_detection(scores_adv, labels_detec)

    # Average performance over the test folds
    auc_avg = np.mean(auc_scores)
    avg_precision_avg = np.mean(avg_precision_scores)
    pauc_avg = np.mean(pauc_scores, axis=0)
    tpr_avg = np.mean(tpr_scores, axis=0)
    fpr_avg = np.mean(fpr_scores, axis=0)

    # Save the performance metrics to a file


if __name__ == '__main__':
    main()
