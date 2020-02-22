"""
Main script for running the adversarial and OOD detection experiments.
"""
from __future__ import absolute_import, division, print_function
import sys
import argparse
import os
import random
import numpy as np
import torch
from torchvision import datasets, transforms
from nets.mnist import *
# from nets.cifar10 import *
from nets.svhn import *
from nets.resnet import *
from helpers.constants import *
from helpers.utils import (
    load_model_checkpoint,
    save_model_checkpoint,
    convert_to_loader,
    load_numpy_data,
    get_path_dr_models,
    get_clean_data_path,
    get_adversarial_data_path,
    get_output_path,
    list_all_adversarial_subdirs,
    check_label_mismatch,
    metrics_varying_positive_class_proportion
)
from helpers.dimension_reduction_methods import load_dimension_reduction_models
from detectors.detector_odds_are_odd import (
    get_samples_as_ndarray,
    get_wcls,
    return_data,
    fit_odds_are_odd,
    detect_odds_are_odd
)
from detectors.detector_lid_paper import DetectorLID
from detectors.detector_proposed import DetectorLayerStatistics, extract_layer_embeddings
from detectors.detector_deep_knn import DeepKNN
from detectors.detector_trust_score import TrustScore


# Parameters that were used for the attack data generation
ATTACK_PARAMS = {
    'stepsize': 0.05,
    'confidence': 0,
    'epsilon': EPSILON_VALUES[0],
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


def helper_layer_embeddings(model, device, data_loader, method, labels_orig):
    layer_embeddings, labels, labels_pred, _ = extract_layer_embeddings(
        model, device, data_loader, method=method
    )
    # NOTE: `labels` returned by this function should be the same as the `labels_orig`
    if not np.array_equal(labels_orig, labels):
        raise ValueError("Class labels returned by 'extract_layer_embeddings' is different from the original "
                         "labels.")

    return layer_embeddings, labels_pred


def get_config_trust_score(modelfile_dim_reduc, layer_type):
    config_trust_score = dict()
    # defines the `1 - alpha` density level set
    config_trust_score['alpha'] = 0.0
    # number of neighbors; set to 10 in the paper
    config_trust_score['n_neighbors'] = None

    if layer_type == 'input':
        layer_index = 0
    elif layer_type == 'logit':
        layer_index = -1
    elif layer_type == 'fc_prelogit':
        layer_index = -2
    else:
        raise ValueError("Unknown layer type '{}'".format(layer_type))

    config_trust_score['layer'] = layer_index
    model_dim_reduc = None
    if modelfile_dim_reduc:
        models = load_dimension_reduction_models(modelfile_dim_reduc)
        model_dim_reduc = models[layer_index]

    config_trust_score['model_dr'] = model_dim_reduc
    return config_trust_score


def get_config_deep_knn():
    return {'n_neighbors': None,    # can be set to other values used in the paper
            'skip_dim_reduction': False}


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256, help='batch size of evaluation')
    parser.add_argument('--model-type', '-m', choices=['mnist', 'cifar10', 'svhn'], default='mnist',
                        help='model type or name of the dataset')
    parser.add_argument('--detection-method', '--dm', choices=DETECTION_METHODS, default='proposed',
                        help="Detection method to run. Choices are: {}".format(', '.join(DETECTION_METHODS)))
    parser.add_argument('--test-statistic', '--ts', choices=TEST_STATS_SUPPORTED, default='multinomial',
                        help="Test statistic to calculate at the layers for the proposed method. Choices are: {}".
                        format(', '.join(TEST_STATS_SUPPORTED)))
    parser.add_argument('--layer-trust-score', '--lts', choices=LAYERS_TRUST_SCORE, default='input',
                        help="Which layer to use for the trust score calculation. Choices are: {}".
                        format(', '.join(LAYERS_TRUST_SCORE)))
    parser.add_argument(
        '--use-top-ranked', '--utr', action='store_true', default=False,
        help='Option that enables only top-ranked test statistics from the layers to be used for the proposed method'
    )
    parser.add_argument(
        '--num-top-ranked', '--ntr', type=int, default=NUM_TOP_RANKED,
        help='If the option --use-top-ranked is provided, this option specifies the number of top-ranked test '
             'statistics to be used by the proposed method'
    )
    parser.add_argument('--modelfile-dim-reduc', '--mdr', default='',
                        help='Path to the saved dimension reduction model file. Specify only if the default path '
                             'needs to be changed.')
    parser.add_argument('--output-dir', '-o', default='', help='directory path for saving the results of detection')
    parser.add_argument('--adv-attack', '--aa', choices=['FGSM', 'PGD', 'CW'], default='PGD',
                        help='type of adversarial attack')
    # parser.add_argument('--p-norm', '-p', choices=['0', '2', 'inf'], default='inf',
    #                     help="p norm for the adversarial attack; options are '0', '2' and 'inf'")
    parser.add_argument('--mixed-attack', '--ma', action='store_true', default=False,
                        help='Use option to enable a mixed attack strategy with multiple methods in '
                             'different proportions')
    parser.add_argument('--num-folds', '--nf', type=int, default=CROSS_VAL_SIZE,
                        help='number of cross-validation folds')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu', type=str, default='2', help='which gpus to execute code on')
    parser.add_argument('--n-jobs', type=int, default=8, help='number of parallel jobs to use for multiprocessing')
    parser.add_argument('--seed', '-s', type=int, default=SEED_DEFAULT, help='seed for random number generation')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs_loader = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    random.seed(args.seed)

    # Output directory
    if not args.output_dir:
        base_dir = get_output_path(args.model_type)
        output_dir = os.path.join(base_dir, 'detection')
    else:
        output_dir = args.output_dir

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Method name for results and plots
    method_name = METHOD_NAME_MAP[args.detection_method]

    # Dimensionality reduction to the layer embeddings is applied only for methods in certain configurations
    apply_dim_reduc = False
    if args.detection_method == 'proposed':
        # Append the test statistic type to the method name
        if not args.use_top_ranked:
            method_name = '{}_{}_all'.format(method_name, args.test_statistic)
        else:
            method_name = '{}_{}_top{:d}'.format(method_name, args.test_statistic, args.num_top_ranked)

        # Dimension reduction is not applied when the test statistic is 'lid' or 'lle'
        if args.test_statistic == 'multinomial':
            apply_dim_reduc = True

    elif args.detection_method == 'trust':
        # Append the layer name to the method name
        method_name = '{}_{}'.format(method_name, args.layer_trust_score)
        # Dimension reduction is not applied to the logit layer
        if args.layer_trust_score != 'logit':
            apply_dim_reduc = True

    elif args.detection_method == 'dknn':
        apply_dim_reduc = True

    # Model file for dimension reduction, if required
    modelfile_dim_reduc = None
    if apply_dim_reduc:
        if args.modelfile_dim_reduc:
            modelfile_dim_reduc = args.modelfile_dim_reduc
        else:
            # Default path to the dimension reduction model file
            modelfile_dim_reduc = get_path_dr_models(args.model_type)

        if not os.path.isfile(modelfile_dim_reduc):
            raise ValueError("Model file for dimension reduction is required, but does not exist: {}".
                             format(modelfile_dim_reduc))

    config_trust_score = dict()
    if args.detection_method == 'trust':
        # Get the layer index and load the layer-specific dimensionality reduction model for the trust score
        config_trust_score = get_config_trust_score(modelfile_dim_reduc, args.layer_trust_score)

    # Data loader and pre-trained DNN model corresponding to the dataset
    data_path = DATA_PATH
    if args.model_type == 'mnist':
        '''
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(*NORMALIZE_IMAGES['mnist'])]
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train=False, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs_loader
        )
        '''
        num_classes = 10
        model = MNIST().to(device)
        model = load_model_checkpoint(model, args.model_type)

    elif args.model_type == 'cifar10':
        '''
        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(*NORMALIZE_IMAGES['cifar10'])]
        )
        testset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, **kwargs_loader)
        '''
        num_classes = 10
        model = ResNet34().to(device)
        model = load_model_checkpoint(model, args.model_type)

    elif args.model_type == 'svhn':
        '''
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(*NORMALIZE_IMAGES['svhn'])]
        )
        testset = datasets.SVHN(root=data_path, split='test', download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, **kwargs_loader)
        '''
        num_classes = 10
        model = SVHN().to(device)
        model = load_model_checkpoint(model, args.model_type)

    else:
        raise ValueError("'{}' is not a valid model type".format(args.model_type))

    # Set model in evaluation mode
    model.eval()

    # Check if the numpy data directory exists
    d = os.path.join(NUMPY_DATA_PATH, args.model_type)
    if not os.path.isdir(d):
        raise ValueError("Directory for the numpy data files not found: {}".format(d))

    attack_params_list = [
        ('stepsize', ATTACK_PARAMS['stepsize']),
        ('confidence', ATTACK_PARAMS['confidence']),
        ('epsilon', ATTACK_PARAMS['epsilon']),
        ('maxiterations', ATTACK_PARAMS['maxiterations']),
        ('iterations', ATTACK_PARAMS['iterations']),
        ('maxepsilon', ATTACK_PARAMS['maxepsilon']),
        ('pnorm', ATTACK_NORM_MAP[args.adv_attack])
    ]

    # Cross-validation
    scores_folds = []
    labels_folds = []
    for i in range(args.num_folds):
        print("\nProcessing cross-validation fold {:d}:".format(i + 1))
        # Load the saved clean numpy data from this fold
        numpy_save_path = get_clean_data_path(args.model_type, i + 1)
        data_tr, labels_tr, data_te, labels_te = load_numpy_data(numpy_save_path, adversarial=False)

        # Data loader for the train fold
        train_fold_loader = convert_to_loader(data_tr, labels_tr, batch_size=args.batch_size, device=device)
        # Data loader for the test fold
        test_fold_loader = convert_to_loader(data_te, labels_te, batch_size=args.batch_size, device=device)

        print("\nCalculating the layer embeddings and DNN predictions for the clean train data split:")
        layer_embeddings_tr, labels_pred_tr = helper_layer_embeddings(
            model, device, train_fold_loader, args.detection_method, labels_tr
        )
        print("\nCalculating the layer embeddings and DNN predictions for the clean test data split:")
        layer_embeddings_te, labels_pred_te = helper_layer_embeddings(
            model, device, test_fold_loader, args.detection_method, labels_te
        )

        # Load the saved adversarial numpy data generated from this training and test fold
        # numpy_save_path = get_adversarial_data_path(args.model_type, i + 1, args.adv_attack, attack_params_list)
        numpy_save_path = list_all_adversarial_subdirs(args.model_type, i + 1, args.adv_attack)[0]
        data_tr_adv, labels_tr_adv, data_te_adv, labels_te_adv = load_numpy_data(numpy_save_path, adversarial=True)

        num_adv_tr = labels_tr_adv.shape[0]
        num_adv_te = labels_te_adv.shape[0]
        print("\nTrain fold: number of clean samples = {:d}, number of adversarial samples = {:d}, % of adversarial "
              "samples = {:.4f}".format(labels_tr.shape[0], num_adv_tr, (100. * num_adv_tr) / labels_tr.shape[0]))
        print("Test fold: number of clean samples = {:d}, number of adversarial samples = {:d}, % of adversarial "
              "samples = {:.4f}".format(labels_te.shape[0], num_adv_te, (100. * num_adv_te) / labels_te.shape[0]))
        # Adversarial data loader for the train fold
        adv_train_fold_loader = convert_to_loader(data_tr_adv, labels_tr_adv, batch_size=args.batch_size,
                                                  device=device)
        # Adversarial data loader for the test fold
        adv_test_fold_loader = convert_to_loader(data_te_adv, labels_te_adv, batch_size=args.batch_size,
                                                 device=device)
        if args.detection_method == 'lid':
            # Needed only for the LID method
            print("\nCalculating the layer embeddings and DNN predictions for the adversarial train data split:")
            layer_embeddings_tr_adv, labels_pred_tr_adv = helper_layer_embeddings(
                model, device, adv_train_fold_loader, args.detection_method, labels_tr_adv
            )
            check_label_mismatch(labels_tr_adv, labels_pred_tr_adv)

        print("\nCalculating the layer embeddings and DNN predictions for the adversarial test data split:")
        layer_embeddings_te_adv, labels_pred_te_adv = helper_layer_embeddings(
            model, device, adv_test_fold_loader, args.detection_method, labels_te_adv
        )
        check_label_mismatch(labels_te_adv, labels_pred_te_adv)

        # TODO: Load noisy data from the saved numpy files, create data loaders, and extract layer embeddings

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
            _ = model_det.fit(layer_embeddings_tr, layer_embeddings_tr_adv,
                              layer_embeddings_noisy=layer_embeddings_tr_noisy)
            # Scores on clean data from the test fold
            scores_adv1 = model_det.score(layer_embeddings_te)

            # Scores on adversarial data from the test fold
            scores_adv2 = model_det.score(layer_embeddings_te_adv)

            scores_adv = np.concatenate([scores_adv1, scores_adv2])

        elif args.detection_method == 'proposed':
            det_model = DetectorLayerStatistics(
                layer_statistic=args.test_statistic,
                use_top_ranked=args.use_top_ranked,
                num_top_ranked=args.num_top_ranked,
                skip_dim_reduction=(not apply_dim_reduc),
                model_file_dim_reduction=modelfile_dim_reduc,
                n_jobs=args.n_jobs,
                seed_rng=args.seed
            )
            # Fit the detector on clean data from the training fold
            _ = det_model.fit(layer_embeddings_tr, labels_tr, labels_pred_tr)

            # Scores on clean data from the test fold
            scores_adv1, scores_ood1 = det_model.score(layer_embeddings_te, labels_pred_te)

            # Scores on adversarial data from the test fold
            scores_adv2, scores_ood2 = det_model.score(layer_embeddings_te_adv, labels_pred_te_adv)

            # scores_adv = np.concatenate([scores_adv1, scores_adv2])
            scores_adv = np.concatenate([scores_ood1, scores_ood2])

        elif args.detection_method == 'dknn':
            config_dknn = get_config_deep_knn()
            det_model = DeepKNN(
                n_neighbors=config_dknn['n_neighbors'],
                skip_dim_reduction=config_dknn['skip_dim_reduction'],
                model_file_dim_reduction=modelfile_dim_reduc,
                n_jobs=args.n_jobs,
                seed_rng=args.seed
            )
            # Fit the detector on clean data from the training fold
            _ = det_model.fit(layer_embeddings_tr, labels_tr)

            # Scores on clean data from the test fold
            scores_adv1, labels_pred_dknn1 = det_model.score(layer_embeddings_te)

            # Scores on adversarial data from the test fold
            scores_adv2, labels_pred_dknn2 = det_model.score(layer_embeddings_te_adv)

            scores_adv = np.concatenate([scores_adv1, scores_adv2])
            # labels_pred_dknn = np.concatenate([labels_pred_dknn1, labels_pred_dknn2])

        elif args.detection_method == 'trust':
            ind_layer = config_trust_score['layer']
            det_model = TrustScore(
                alpha=config_trust_score['alpha'],
                n_neighbors=config_trust_score['n_neighbors'],
                skip_dim_reduction=(not apply_dim_reduc),
                model_dim_reduction=config_trust_score['model_dr'],
                n_jobs=args.n_jobs,
                seed_rng=args.seed
            )
            # Fit the detector on clean data from the training fold
            _ = det_model.fit(layer_embeddings_tr[ind_layer], labels_tr, labels_pred_tr)

            # Scores on clean data from the test fold
            scores_adv1 = det_model.score(layer_embeddings_te[ind_layer], labels_pred_te)

            # Scores on adversarial data from the test fold
            scores_adv2 = det_model.score(layer_embeddings_te_adv[ind_layer], labels_pred_te_adv)

            scores_adv = np.concatenate([scores_adv1, scores_adv2])
        else:
            raise ValueError("Unknown detection method name '{}'".format(args.detection_method))

        scores_folds.append(scores_adv)
        labels_folds.append(labels_detec)

    print("\nCalculating performance metrics for different proportion of attack samples:")
    fname = os.path.join(output_dir, 'detection_metrics_{}.pkl'.format(method_name))
    results_dict = metrics_varying_positive_class_proportion(scores_folds, labels_folds, n_jobs=args.n_jobs,
                                                             output_file=fname)
    print("Performance metrics saved to the file: {}".format(fname))


if __name__ == '__main__':
    main()
