"""
Main script for running the adversarial and OOD detection experiments.
"""
from __future__ import absolute_import, division, print_function
import sys
import argparse
import os
import time
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
    get_data_bounds,
    load_adversarial_data,
    load_noisy_data,
    get_path_dr_models,
    get_clean_data_path,
    get_noisy_data_path,
    get_adversarial_data_path,
    get_output_path,
    list_all_adversarial_subdirs,
    check_label_mismatch,
    metrics_varying_positive_class_proportion
)
from helpers.dimension_reduction_methods import load_dimension_reduction_models
from detectors.detector_odds_are_odd import (
    get_wcls,
    return_data,
    fit_odds_are_odd,
    detect_odds_are_odd
)
from detectors.detector_lid_paper import DetectorLID, DetectorLIDClassCond, DetectorLIDBatch
from detectors.detector_proposed import DetectorLayerStatistics, extract_layer_embeddings
from detectors.detector_deep_knn import DeepKNN
from detectors.detector_trust_score import TrustScore
try:
    import cPickle as pickle
except:
    import pickle


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

# Set to False before releasing the code
PATH_HACK = True


def helper_layer_embeddings(model, device, data_loader, method, labels_orig):
    layer_embeddings, labels, labels_pred, _ = extract_layer_embeddings(
        model, device, data_loader, method=method
    )
    # NOTE: `labels` returned by this function should be the same as the `labels_orig`
    if not np.array_equal(labels_orig, labels):
        raise ValueError("Class labels returned by 'extract_layer_embeddings' is different from the original "
                         "labels.")

    return layer_embeddings, labels_pred


def get_config_trust_score(model_dim_reduc, layer_type, n_neighbors):
    config_trust_score = dict()
    # defines the `1 - alpha` density level set
    config_trust_score['alpha'] = 0.0
    # number of neighbors; set to 10 in the paper
    config_trust_score['n_neighbors'] = n_neighbors

    if layer_type == 'input':
        layer_index = 0
    elif layer_type == 'logit':
        layer_index = -1
    elif layer_type == 'prelogit':
        layer_index = -2
    else:
        raise ValueError("Unknown layer type '{}'".format(layer_type))

    config_trust_score['layer'] = layer_index

    # Dimension reduction model for the specified layer
    if model_dim_reduc:
        config_trust_score['model_dr'] = model_dim_reduc[layer_index]
    else:
        config_trust_score['model_dr'] = None

    return config_trust_score


def save_checkpoint(scores_folds, labels_folds, models_folds, output_dir, method_name, save_detec_model):
    # Save the scores and detection labels from the cross-validation folds to a pickle file
    fname = os.path.join(output_dir, 'scores_{}.pkl'.format(method_name))
    tmp = {'scores_folds': scores_folds, 'labels_folds': labels_folds}
    with open(fname, 'wb') as fp:
        pickle.dump(tmp, fp)

    if save_detec_model and models_folds:
        # Save the detection models from the cross-validation folds to a pickle file
        fname = os.path.join(output_dir, 'models_{}.pkl'.format(method_name))
        with open(fname, 'wb') as fp:
            pickle.dump(models_folds, fp)


def load_checkpoint(output_dir, method_name, save_detec_model):
    fname = os.path.join(output_dir, 'scores_{}.pkl'.format(method_name))
    with open(fname, 'rb') as fp:
        tmp = pickle.load(fp)

    scores_folds = tmp['scores_folds']
    labels_folds = tmp['labels_folds']
    # Load the models if required
    if save_detec_model:
        fname = os.path.join(output_dir, 'models_{}.pkl'.format(method_name))
        with open(fname, 'rb') as fp:
            models_folds = pickle.load(fp)
    else:
        models_folds = []

    n_folds = len(scores_folds)
    assert len(labels_folds) == n_folds, "'scores_folds' and 'labels_folds' do not have the same length"
    if models_folds:
        assert len(models_folds) == n_folds, "'models_folds' and 'scores_folds' do not have the same length"

    return scores_folds, labels_folds, models_folds, n_folds


def load_adversarial_wrapper(i, model_type, adv_attack, max_attack_prop, num_clean_te, index=0):
    # Helper function to load adversarial data from the saved numpy files for cross-validation fold `i`
    if adv_attack != CUSTOM_ATTACK:
        # Load the saved adversarial numpy data generated from this training and test fold
        # numpy_save_path = get_adversarial_data_path(model_type, i + 1, adv_attack, attack_params_list)
        numpy_save_path = list_all_adversarial_subdirs(model_type, i + 1, adv_attack)[index]
        if PATH_HACK:
            # Temporary hack to use backup data directory
            numpy_save_path = numpy_save_path.replace('varun', 'jayaram', 1)

        # Maximum number of adversarial samples in the test fold
        max_num_adv = int(np.ceil((max_attack_prop / (1. - max_attack_prop)) * num_clean_te))
        data_tr_adv, labels_tr_adv, data_te_adv, labels_te_adv = load_adversarial_data(
            numpy_save_path,
            max_n_test=max_num_adv,
            sampling_type='ranked_by_norm',
            norm_type=ATTACK_NORM_MAP.get(adv_attack, '2')
        )
    else:
        # Custom attack
        numpy_save_path = list_all_adversarial_subdirs(model_type, i + 1, adv_attack, check_subdirectories=False)[0]
        if PATH_HACK:
            # Temporary hack to use backup data directory
            numpy_save_path = numpy_save_path.replace('jayaram', 'varun', 1)

        # Adversarial inputs from the train and test fold
        data_te = np.load(os.path.join(numpy_save_path, "data_te_adv.npy"))
        # Predicted (mis-classified) labels
        labels_pred_te = np.load(os.path.join(numpy_save_path, "labels_te_adv.npy"))
        # Labels of the original inputs from which the adversarial inputs were created
        labels_te = np.load(os.path.join(numpy_save_path, "labels_te_clean.npy"))
        # Norm of perturbations
        norm_perturb = np.load(os.path.join(numpy_save_path, 'norm_perturb.npy'))
        # Mask indicating which samples are adversarial
        mask = np.load(os.path.join(numpy_save_path, "is_adver.npy"))
        # Adversarial samples with very large perturbation are excluded
        mask_norm = norm_perturb <= np.percentile(norm_perturb[mask], 95.)
        mask_incl = np.logical_and(mask, mask_norm)

        data_te_adv = data_te[mask_incl, :]
        labels_te_adv = labels_te[mask_incl]
        # Check if the original and predicted labels are all different for the adversarial inputs
        check_label_mismatch(labels_te_adv, labels_pred_te[mask_incl])

        # We don't generate adversarial samples from the train fold for the custom attack.
        # Returning the train fold data from Carlini-Wagner attack instead. This is used only by the supervised
        # detection methods
        data_tr_adv, labels_tr_adv, _, _ = load_adversarial_wrapper(
            i, model_type, 'CW', max_attack_prop, num_clean_te
        )
        # data_tr_adv = data_te_adv
        # labels_tr_adv = labels_te_adv
    
    return data_tr_adv, labels_tr_adv, data_te_adv, labels_te_adv


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256, help='batch size of evaluation')
    parser.add_argument('--model-type', '-m', choices=['mnist', 'cifar10', 'svhn'], default='mnist',
                        help='model type or name of the dataset')
    parser.add_argument('--detection-method', '--dm', choices=DETECTION_METHODS, default='proposed',
                        help="Detection method to run. Choices are: {}".format(', '.join(DETECTION_METHODS)))
    parser.add_argument('--resume-from-ckpt', action='store_true', default=False,
                        help='Use this option to load results and resume from a previous partially completed run. '
                             'Cross-validation folds that were completed earlier will be skipped in the current run.')
    parser.add_argument('--save-detec-model', action='store_true', default=False,
                        help='Use this option to save the list of detection models from the CV folds to a pickle '
                             'file. Note that the files tend to large in size.')
    ################ Optional arguments for the proposed method
    parser.add_argument('--test-statistic', '--ts', choices=TEST_STATS_SUPPORTED, default='multinomial',
                        help="Test statistic to calculate at the layers for the proposed method. Choices are: {}".
                        format(', '.join(TEST_STATS_SUPPORTED)))
    parser.add_argument('--score-type', '--st', choices=SCORE_TYPES, default='pvalue',
                        help="Score type to use for the proposed method. Choices are: {}".
                        format(', '.join(SCORE_TYPES)))
    parser.add_argument('--pvalue-fusion', '--pf', choices=['harmonic_mean', 'fisher'], default='harmonic_mean',
                        help="Name of the method to use for combining p-values from multiple layers for the "
                             "proposed method. Choices are: 'harmonic_mean' and 'fisher'")
    parser.add_argument('--ood-detection', '--ood', action='store_true', default=False,
                        help="Option that enables out-of-distribution detection instead of adversarial detection "
                             "for the proposed method")
    parser.add_argument(
        '--use-top-ranked', '--utr', action='store_true', default=False,
        help="Option that enables the proposed method to use only the top-ranked (by p-values) test statistics for "
             "detection. The number of test statistics is specified through the option '--num-layers'"
    )
    parser.add_argument(
        '--use-deep-layers', '--udl', action='store_true', default=False,
        help="Option that enables the proposed method to use only a given number of last few layers of the DNN. "
             "The number of layers is specified through the option '--num-layers'"
    )
    parser.add_argument(
        '--num-layers', '--nl', type=int, default=NUM_TOP_RANKED,
        help="If the option '--use-top-ranked' or '--use-deep-layers' is provided, this option specifies the number "
             "of layers or test statistics to be used by the proposed method"
    )
    parser.add_argument(
        '--combine-classes', '--cc', action='store_true', default=False,
        help="Option that allows low probability classes to be automatically combined into one group for the "
             "multinomial test statistic used with the proposed method"
    )
    ################ Optional arguments for the proposed method
    parser.add_argument('--layer-trust-score', '--lts', choices=LAYERS_TRUST_SCORE, default='input',
                        help="Which layer to use for the trust score calculation. Choices are: {}".
                        format(', '.join(LAYERS_TRUST_SCORE)))
    parser.add_argument('--batch-lid', action='store_true', default=False,
                        help='Use this option to enable batched, faster version of the LID detector')
    parser.add_argument('--num-neighbors', '--nn', type=int, default=-1,
                        help='Number of nearest neighbors (if applicable to the method). By default, this is set '
                             'to be a power of the number of samples (n): n^{:.1f}'.format(NEIGHBORHOOD_CONST))
    parser.add_argument('--modelfile-dim-reduc', '--mdr', default='',
                        help='Path to the saved dimension reduction model file. Specify only if the default path '
                             'needs to be changed.')
    parser.add_argument('--output-dir', '-o', default='', help='directory path for saving the results of detection')
    parser.add_argument('--adv-attack', '--aa', choices=['FGSM', 'PGD', 'CW', CUSTOM_ATTACK], default='PGD',
                        help='type of adversarial attack')
    # parser.add_argument('--p-norm', '-p', choices=['0', '2', 'inf'], default='inf',
    #                     help="p norm for the adversarial attack; options are '0', '2' and 'inf'")
    parser.add_argument('--max-attack-prop', '--map', type=float, default=0.5,
                        help="Maximum proportion of attack samples in the test fold. Should be a value in (0, 1]")
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

    if args.use_top_ranked and args.use_deep_layers:
        raise ValueError("Cannot provide both command line options '--use-top-ranked' and '--use-deep-layers'. "
                         "Specify only one of them.")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs_loader = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    random.seed(args.seed)

    # Number of neighbors
    n_neighbors = args.num_neighbors
    if n_neighbors <= 0:
        n_neighbors = None

    # Output directory
    if not args.output_dir:
        base_dir = get_output_path(args.model_type)
        output_dir = os.path.join(base_dir, 'prediction')
    else:
        output_dir = args.output_dir

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Method name for results and plots
    method_name = METHOD_NAME_MAP[args.detection_method]

    # Dimensionality reduction to the layer embeddings is applied only for methods in certain configurations
    apply_dim_reduc = False
    if args.detection_method == 'proposed':
        # Name string for the proposed method based on the input configuration
        # Score type suffix in the method name
        st = '{:.4s}'.format(args.score_type)
        if args.score_type == 'pvalue':
            if args.pvalue_fusion == 'harmonic_mean':
                st += '_hmp'
            if args.pvalue_fusion == 'fisher':
                st += '_fis'

        if not args.ood_detection:
            method_name = '{:.5s}_{:.5s}_{}_adv'.format(method_name, args.test_statistic, st)
        else:
            method_name = '{:.5s}_{:.5s}_{}_ood'.format(method_name, args.test_statistic, st)

        if args.use_top_ranked:
            method_name = '{}_top{:d}'.format(method_name, args.num_layers)
        elif args.use_deep_layers:
            method_name = '{}_last{:d}'.format(method_name, args.num_layers)

        # If `n_neighbors` is specified, append that value to the name string
        if n_neighbors is not None:
            method_name = '{}_k{:d}'.format(method_name, n_neighbors)

        apply_dim_reduc = True
    
    elif args.detection_method == 'dknn':
        apply_dim_reduc = True
        # If `n_neighbors` is specified, append that value to the name string
        if n_neighbors is not None:
            method_name = '{}_k{:d}'.format(method_name, n_neighbors)
    
    # Model file for dimension reduction, if required
    model_dim_reduc = None
    if apply_dim_reduc:
        if args.modelfile_dim_reduc:
            fname = args.modelfile_dim_reduc
        else:
            # Path to the dimension reduction model file
            fname = get_path_dr_models(args.model_type, args.detection_method, test_statistic=args.test_statistic)

        if not os.path.isfile(fname):
            raise ValueError("Model file for dimension reduction is required, but does not exist: {}".format(fname))
        else:
            # Load the dimension reduction models for each layer from the pickle file
            model_dim_reduc = load_dimension_reduction_models(fname)
    
    config_trust_score = dict()
    
    # Data loader and pre-trained DNN model corresponding to the dataset
    data_path = DATA_PATH
    if args.model_type == 'mnist':
        num_classes = 10
        model = MNIST().to(device)
        model = load_model_checkpoint(model, args.model_type)

    elif args.model_type == 'cifar10':
        num_classes = 10
        model = ResNet34().to(device)
        model = load_model_checkpoint(model, args.model_type)

    elif args.model_type == 'svhn':
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

    # Not used
    attack_params_list = [
        ('stepsize', ATTACK_PARAMS['stepsize']),
        ('confidence', ATTACK_PARAMS['confidence']),
        ('epsilon', ATTACK_PARAMS['epsilon']),
        ('maxiterations', ATTACK_PARAMS['maxiterations']),
        ('iterations', ATTACK_PARAMS['iterations']),
        ('maxepsilon', ATTACK_PARAMS['maxepsilon']),
        ('pnorm', ATTACK_NORM_MAP[args.adv_attack])
    ]

    # Initialization
    if args.resume_from_ckpt:
        scores_folds, labels_folds, models_folds, init_fold = load_checkpoint(output_dir, method_name,
                                                                              args.save_detec_model)
        print("Loading saved results from a previous run. Completed {:d} fold(s). Resuming from fold {:d}.".
              format(init_fold, init_fold + 1))
    else:
        scores_folds = []
        labels_folds = []
        models_folds = []
        init_fold = 0

    ti = time.time()

    clean_labels = []
    
    # Cross-validation
    for i in range(init_fold, args.num_folds):
        print("\nProcessing cross-validation fold {:d}:".format(i + 1))
        # Load the saved clean numpy data from this fold
        numpy_save_path = get_clean_data_path(args.model_type, i + 1)
        if PATH_HACK:
            # Temporary hack to use backup data directory
            numpy_save_path = numpy_save_path.replace('varun', 'jayaram', 1)

        data_tr, labels_tr, data_te, labels_te = load_numpy_data(numpy_save_path)
        num_clean_tr = labels_tr.shape[0]
        num_clean_te = labels_te.shape[0]
        
        clean_labels.append(data_te)

        # Data loader for the train fold
        train_fold_loader = convert_to_loader(data_tr, labels_tr, batch_size=args.batch_size, device=device)
        # Data loader for the test fold
        test_fold_loader = convert_to_loader(data_te, labels_te, batch_size=args.batch_size, device=device)

        # Get the range of values in the data array
        bounds = get_data_bounds(np.concatenate([data_tr, data_te], axis=0))

        print("\nCalculating the layer embeddings and DNN predictions for the clean train data split:")
        layer_embeddings_tr, labels_pred_tr = helper_layer_embeddings(
            model, device, train_fold_loader, args.detection_method, labels_tr
        )
        print("\nCalculating the layer embeddings and DNN predictions for the clean test data split:")
        layer_embeddings_te, labels_pred_te = helper_layer_embeddings(
            model, device, test_fold_loader, args.detection_method, labels_te
        )
        
        train_fold_loader = None
        if args.detection_method != 'odds':
            test_fold_loader = None
        
        if args.adv_attack == 'CW':
            max_index = 4
        elif args.adv_attack == 'PGD':
            max_index = 12
        else:
            max_index = 1
        
        labels_adv_folds = []

        for index in range(0, max_index):
            # Load the saved adversarial numpy data generated from this training and test fold
            data_tr_adv, labels_tr_adv, data_te_adv, labels_te_adv = load_adversarial_wrapper(
                i, args.model_type, args.adv_attack, args.max_attack_prop, num_clean_te, index
            )
            num_adv_tr = labels_tr_adv.shape[0]
            num_adv_te = labels_te_adv.shape[0]
            
            print("\nTrain fold: number of clean samples = {:d}, number of adversarial samples = {:d}, % of adversarial "
                "samples = {:.4f}".format(num_clean_tr, num_adv_tr, (100. * num_adv_tr) / (num_clean_tr + num_adv_tr)))
            print("Test fold: number of clean samples = {:d}, number of adversarial samples = {:d}, % of adversarial "
                "samples = {:.4f}".format(num_clean_te, num_adv_te, (100. * num_adv_te) / (num_clean_te + num_adv_te)))

            # Adversarial data loader for the train fold
            adv_train_fold_loader = convert_to_loader(data_tr_adv, labels_tr_adv, batch_size=args.batch_size,
                                                  device=device)
            # Adversarial data loader for the test fold
            adv_test_fold_loader = convert_to_loader(data_te_adv, labels_te_adv, batch_size=args.batch_size,
                                                 device=device)

            print("\nCalculating the layer embeddings and DNN predictions for the adversarial test data split:")
            layer_embeddings_te_adv, labels_pred_te_adv = helper_layer_embeddings(
                model, device, adv_test_fold_loader, args.detection_method, labels_te_adv
            )
            check_label_mismatch(labels_te_adv, labels_pred_te_adv)
            adv_train_fold_loader = None
    
            if args.detection_method != 'odds':
                adv_test_fold_loader = None

            # Detection labels (0 denoting clean and 1 adversarial)
            labels_detec = np.concatenate([np.zeros(labels_pred_te.shape[0], dtype=np.int),
                                       np.ones(labels_pred_te_adv.shape[0], dtype=np.int)])
            
            if args.detection_method == 'proposed':
                nl = len(layer_embeddings_tr)
                st_ind = 0
                if args.use_deep_layers:
                    if args.num_layers > nl:
                        print("WARNING: number of layers specified using the option '--num-layers' exceeds the number "
                          "of layers in the model. Using all the layers.")
                        st_ind = 0
                    else:
                        st_ind = nl - args.num_layers
                        print("Using only the last {:d} layer embeddings from the {:d} layers for the proposed method.".
                          format(args.num_layers, nl))

                mod_dr = None if (model_dim_reduc is None) else model_dim_reduc[st_ind:]
                det_model = DetectorLayerStatistics(
                    layer_statistic=args.test_statistic,
                    score_type=args.score_type,
                    ood_detection=args.ood_detection,
                    pvalue_fusion=args.pvalue_fusion,
                    use_top_ranked=args.use_top_ranked,
                    num_top_ranked=args.num_layers,
                    skip_dim_reduction=(not apply_dim_reduc),
                    model_dim_reduction=mod_dr,
                    n_neighbors=n_neighbors,
                    n_jobs=args.n_jobs,
                    seed_rng=args.seed
                )
                # Fit the detector on clean data from the training fold
                if args.combine_classes and (args.test_statistic == 'multinomial'):
                    _ = det_model.fit(layer_embeddings_tr[st_ind:], labels_tr, labels_pred_tr,
                                  combine_low_proba_classes=True)
                else:
                    _ = det_model.fit(layer_embeddings_tr[st_ind:], labels_tr, labels_pred_tr)

                # Scores on clean data from the test fold
                scores_adv1, corrected_classes_clean = det_model.score(layer_embeddings_te[st_ind:], labels_pred_te, return_corrected_predictions=True, test_layer_pairs=True)

                # Scores on adversarial data from the test fold
                scores_adv2, corrected_classes_adv = det_model.score(layer_embeddings_te_adv[st_ind:], labels_pred_te_adv, return_corrected_predictions=True, test_layer_pairs=True)

                scores_adv = np.concatenate([scores_adv1, scores_adv2])
                labels_adv = np.concatenate([corrected_classes_clean, corrected_classes_adv])

            elif args.detection_method == 'dknn':
                det_model = DeepKNN(
                    n_neighbors=n_neighbors,
                    skip_dim_reduction=(not apply_dim_reduc),
                    model_dim_reduction=model_dim_reduc,
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
                labels_adv = np.concatenate([labels_pred_dknn1, labels_pred_dknn2])
                
            
            else:
                raise ValueError("Unknown detection method name '{}'".format(args.detection_method))

            # Sanity check
            if scores_adv.shape[0] != labels_detec.shape[0]:
                raise ValueError(
                    "Detection scores and labels do not have the same length ({:d} != {:d}); method = {}, fold = {:d}".
                        format(scores_adv.shape[0], labels_detec.shape[0], args.detection_method, i + 1)
                )

            labels_adv_folds.append(labels_adv)
        
        scores_folds.append(scores_adv)
        labels_folds.append(labels_adv_folds)
        #saved every fold
        save_checkpoint(scores_folds, labels_folds, models_folds, output_dir, method_name, args.save_detec_model)

    print("\nCalculating performance metrics for different proportion of attack samples:")
    fname = os.path.join(output_dir, 'corrected_accuracies_{}.pkl'.format(method_name))
    results_dict = metrics_for_accuracy(
        labels_folds, clean_labels, output_file=fname
    )
    print("Performance metrics saved to the file: {}".format(fname))
    tf = time.time()
    print("Total time taken: {:.4f} minutes".format((tf - ti) / 60.))

if __name__ == '__main__':
    main()