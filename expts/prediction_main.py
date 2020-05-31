"""
Main script for calculating the corrected (combined) classification performance of the DNN and the proposed method.
"""
from __future__ import absolute_import, division, print_function
import sys
import argparse
import os
import time
import copy
import pickle
from pprint import pprint
import numpy as np
import torch
from nets.mnist import *
from nets.svhn import *
from nets.resnet import *
from helpers.constants import *
from helpers.utils import (
    load_model_checkpoint,
    convert_to_loader,
    load_numpy_data,
    load_adversarial_data,
    get_path_dr_models,
    get_clean_data_path,
    get_noisy_data_path,
    get_adversarial_data_path,
    get_output_path,
    list_all_adversarial_subdirs,
    check_label_mismatch,
    helper_layer_embeddings,
    load_adversarial_wrapper
)
from helpers.dimension_reduction_methods import load_dimension_reduction_models
from detectors.detector_proposed import DetectorLayerStatistics
from detectors.detector_deep_knn import DeepKNN

# Target FPRs for setting thresholds of the detector (1%, 5%, and 10%)
FPRS_TARGET = [0.01, 0.05, 0.1]


def find_score_thresholds(scores_detec, fprs_target):
    thresholds = np.zeros(len(fprs_target))
    for i, fpr in enumerate(fprs_target):
        thresholds[i] = np.percentile(scores_detec, 100 * (1. - fpr), interpolation='higher')

    print("\nScore thresholds: {}".format(', '.join(['{:.4f}'.format(t) for t in thresholds])))
    return thresholds


def combined_classification_performance(scores_detec_folds, thresholds_folds, labels_pred_detec_folds,
                                        labels_pred_dnn_folds, labels_true_folds, fprs_target, output_file=None):
    def _accuracy(preds, labels):
        mask = preds == labels
        return (100. * mask[mask].shape[0]) / labels.shape[0]

    n_folds = len(thresholds_folds)
    n_fprs = len(fprs_target)
    results = {
        'FPR_target': np.array(fprs_target),
        'accuracy_dnn': 0.,     # this accuracy does not depend on the target FPR
        'accuracy_combined': np.zeros(n_fprs)
    }
    for i in range(n_folds):
        # print("Fold {:d}:".format(i + 1))
        # Accuracy of the DNN on this test fold
        results['accuracy_dnn'] += _accuracy(labels_pred_dnn_folds[i], labels_true_folds[i])
        for j, tau in enumerate(thresholds_folds[i]):
            mask = scores_detec_folds[i] >= tau
            # Samples that score above the threshold use the detector's class prediction
            labels_pred = copy.copy(labels_pred_dnn_folds[i])
            labels_pred[mask] = labels_pred_detec_folds[i][mask]
            # Accuracy of the DNN + detector combination
            results['accuracy_combined'][j] += _accuracy(labels_pred, labels_true_folds[i])

    # average accuracy across test folds
    results['accuracy_dnn'] = (1. / n_folds) * results['accuracy_dnn']
    results['accuracy_combined'] = (1. / n_folds) * results['accuracy_combined']

    pprint(results, indent=2)
    # Save the results to a pickle file if required
    if output_file:
        with open(output_file, 'wb') as fp:
            pickle.dump(results, fp)

    return results


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', '-m', choices=['mnist', 'cifar10', 'svhn'], default='mnist',
                        help='model type or name of the dataset')
    parser.add_argument('--detection-method', '--dm', choices=DETECTION_METHODS, default='proposed',
                        help="Detection method to run. Choices are: {}".format(', '.join(DETECTION_METHODS)))
    parser.add_argument('--index-adv', type=int, default=0,
                        help='Index of the adversarial attack parameter to use. This indexes the sorted directories '
                             'containing the adversarial data files from different attack parameters.')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size of evaluation')
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
    parser.add_argument('--num-neighbors', '--nn', type=int, default=-1,
                        help='Number of nearest neighbors (if applicable to the method). By default, this is set '
                             'to be a power of the number of samples (n): n^{:.1f}'.format(NEIGHBORHOOD_CONST))
    parser.add_argument('--modelfile-dim-reduc', '--mdr', default='',
                        help='Path to the saved dimension reduction model file. Specify only if the default path '
                             'needs to be changed.')
    parser.add_argument('--output-dir', '-o', default='', help='directory path for saving the results of detection')
    parser.add_argument('--adv-attack', '--aa', choices=['FGSM', 'PGD', 'CW', CUSTOM_ATTACK, 'none'], default='PGD',
                        help="Type of adversarial attack. Use 'none' to evaluate on clean samples.")
    parser.add_argument('--max-attack-prop', '--map', type=float, default=0.5,
                        help="Maximum proportion of attack samples in the test fold. Should be a value in (0, 1]")
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
        apply_dim_reduc = False
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

    # Data loader and pre-trained DNN model corresponding to the dataset
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

    if args.adv_attack.lower() == 'none':
        evaluate_on_clean = True
    else:
        evaluate_on_clean = False

    # Initialization
    labels_true_folds = []
    labels_pred_dnn_folds = []
    scores_detec_folds = []
    labels_pred_detec_folds = []
    thresholds_folds = []
    ti = time.time()
    # Cross-validation
    for i in range(args.num_folds):
        print("\nProcessing cross-validation fold {:d}:".format(i + 1))
        # Load the saved clean numpy data from this fold
        numpy_save_path = get_clean_data_path(args.model_type, i + 1)
        # Temporary hack to use backup data directory
        numpy_save_path = numpy_save_path.replace('varun', 'jayaram', 1)

        data_tr, labels_tr, data_te, labels_te = load_numpy_data(numpy_save_path)
        num_clean_tr = labels_tr.shape[0]
        num_clean_te = labels_te.shape[0]
        # Data loader for the train and test fold
        train_fold_loader = convert_to_loader(data_tr, labels_tr, dtype_x=torch.float, batch_size=args.batch_size,
                                              device=device)
        test_fold_loader = convert_to_loader(data_te, labels_te, dtype_x=torch.float, batch_size=args.batch_size,
                                             device=device)
        print("\nCalculating the layer embeddings and DNN predictions for the clean train data split:")
        layer_embeddings_tr, labels_pred_tr = helper_layer_embeddings(
            model, device, train_fold_loader, args.detection_method, labels_tr
        )
        print("\nCalculating the layer embeddings and DNN predictions for the clean test data split:")
        layer_embeddings_te, labels_pred_te = helper_layer_embeddings(
            model, device, test_fold_loader, args.detection_method, labels_te
        )
        del train_fold_loader
        del test_fold_loader

        if not evaluate_on_clean:
            # Load the saved adversarial numpy data generated from this training and test fold
            _, _, data_tr_adv, labels_tr_adv, data_te_adv, labels_te_adv = load_adversarial_wrapper(
                i, args.model_type, args.adv_attack, args.max_attack_prop, num_clean_te, index_adv=args.index_adv
            )
            num_adv_tr = labels_tr_adv.shape[0]
            num_adv_te = labels_te_adv.shape[0]
            print("\nTrain fold: number of clean samples = {:d}, number of adversarial samples = {:d}, % of "
                  "adversarial samples = {:.4f}".format(num_clean_tr, num_adv_tr,
                                                        (100. * num_adv_tr) / (num_clean_tr + num_adv_tr)))
            print("Test fold: number of clean samples = {:d}, number of adversarial samples = {:d}, % of adversarial "
                  "samples = {:.4f}".format(num_clean_te, num_adv_te,
                                            (100. * num_adv_te) / (num_clean_te + num_adv_te)))
            # Adversarial data loader for the test fold
            adv_test_fold_loader = convert_to_loader(data_te_adv, labels_te_adv, dtype_x=torch.float,
                                                     batch_size=args.batch_size, device=device)
            print("\nCalculating the layer embeddings and DNN predictions for the adversarial test data split:")
            layer_embeddings_te_adv, labels_pred_te_adv = helper_layer_embeddings(
                model, device, adv_test_fold_loader, args.detection_method, labels_te_adv
            )
            check_label_mismatch(labels_te_adv, labels_pred_te_adv)
            del adv_test_fold_loader

            # True class labels of adversarial samples from this test fold
            labels_true_folds.append(labels_te_adv)
            # Class predictions of the DNN on adversarial samples from this test fold
            labels_pred_dnn_folds.append(labels_pred_te_adv)
            num_expec = num_adv_te
        else:
            print("\nTrain fold: number of clean samples = {:d}".format(num_clean_tr))
            print("Test fold: number of clean samples = {:d}".format(num_clean_te))
            # True class labels of clean samples from this test fold
            labels_true_folds.append(labels_te)
            # Class predictions of the DNN on clean samples from this test fold
            labels_pred_dnn_folds.append(labels_pred_te)
            num_expec = num_clean_te

        # Detection methods
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

            # Find the score thresholds corresponding to the target FPRs using the scores from the clean train
            # fold data
            scores_detec_train = det_model.score(
                layer_embeddings_tr[st_ind:], labels_pred_tr, test_layer_pairs=True, is_train=True
            )
            thresholds = find_score_thresholds(scores_detec_train, FPRS_TARGET)
            if evaluate_on_clean:
                # Scores and class predictions on clean data from the test fold
                scores_detec, labels_pred_detec = det_model.score(
                    layer_embeddings_te[st_ind:], labels_pred_te,
                    return_corrected_predictions=True, test_layer_pairs=True
                )
            else:
                # Scores and class predictions on adversarial data from the test fold
                scores_detec, labels_pred_detec = det_model.score(
                    layer_embeddings_te_adv[st_ind:], labels_pred_te_adv,
                    return_corrected_predictions=True, test_layer_pairs=True
                )

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
            # Find the score thresholds corresponding to the target FPRs using the scores from the clean train
            # fold data
            scores_detec_train = det_model.score(layer_embeddings_tr, is_train=True)
            thresholds = find_score_thresholds(scores_detec_train, FPRS_TARGET)
            if evaluate_on_clean:
                # Scores and class predictions on clean data from the test fold
                scores_detec, labels_pred_detec = det_model.score(layer_embeddings_te)
            else:
                # Scores and class predictions on adversarial data from the test fold
                scores_detec, labels_pred_detec = det_model.score(layer_embeddings_te_adv)

        else:
            raise ValueError("Unknown detection method name '{}'".format(args.detection_method))

        # Sanity check
        if (scores_detec.shape[0] != num_expec) or (labels_pred_detec.shape[0] != num_expec):
            raise ValueError(
                "Detection scores and/or predicted labels do not have the expected length of {:d}; method = {}, "
                "fold = {:d}".format(num_expec, args.detection_method, i + 1)
            )

        scores_detec_folds.append(scores_detec)
        labels_pred_detec_folds.append(labels_pred_detec)
        thresholds_folds.append(thresholds)

    print("\nCalculating the combined classification accuracy of the DNN and detector system:")
    fname = os.path.join(output_dir, 'corrected_accuracies_{}.pkl'.format(method_name))
    results = combined_classification_performance(
        scores_detec_folds, thresholds_folds, labels_pred_detec_folds, labels_pred_dnn_folds, labels_true_folds,
        FPRS_TARGET, output_file=fname
    )
    print("Performance metrics saved to the file: {}".format(fname))
    tf = time.time()
    print("Total time taken: {:.4f} minutes".format((tf - ti) / 60.))


if __name__ == '__main__':
    main()
