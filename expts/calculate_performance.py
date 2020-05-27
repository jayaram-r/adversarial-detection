"""
Read the saved detection scores and labels from a specified method, calculate performance metrics, and save the
performance metrics to a file.

"""
from __future__ import absolute_import, division, print_function
import sys
import argparse
import os
import time
import numpy as np
import torch
from helpers.constants import *
from helpers.utils import (
    load_numpy_data,
    load_adversarial_data,
    get_clean_data_path,
    get_adversarial_data_path,
    list_all_adversarial_subdirs,
    check_label_mismatch,
    metrics_varying_positive_class_proportion,
    load_detector_checkpoint,
    load_adversarial_wrapper
)


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', '-o', required=True,
                        help='directory path with the saved results of detection')
    parser.add_argument('--model-type', '-m', choices=['mnist', 'cifar10', 'svhn'], default='mnist',
                        help='model type or name of the dataset')
    parser.add_argument('--detection-method', '--dm', choices=DETECTION_METHODS, default='proposed',
                        help="Detection method to run. Choices are: {}".format(', '.join(DETECTION_METHODS)))
    parser.add_argument('--x-var', choices=['proportion', 'norm'], default='norm',
                        help="Choice of variable on the x-axis. Options are 'norm' for the perturbation norm, "
                             "and 'proportion' for proportion of adversarial/OOD samples.")
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
    ################ Optional arguments for the proposed method
    parser.add_argument('--layer-trust-score', '--lts', choices=LAYERS_TRUST_SCORE, default='input',
                        help="Which layer to use for the trust score calculation. Choices are: {}".
                        format(', '.join(LAYERS_TRUST_SCORE)))
    parser.add_argument('--num-neighbors', '--nn', type=int, default=-1,
                        help='Number of nearest neighbors (if applicable to the method). By default, this is set '
                             'to be a power of the number of samples (n): n^{:.1f}'.format(NEIGHBORHOOD_CONST))
    parser.add_argument('--adv-attack', '--aa', choices=['FGSM', 'PGD', 'CW', CUSTOM_ATTACK], default='PGD',
                        help='type of adversarial attack')
    parser.add_argument('--max-attack-prop', '--map', type=float, default=0.5,
                        help="Maximum proportion of attack samples in the test fold. Should be a value in (0, 1]")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu', type=str, default='2', help='which gpus to execute code on')
    args = parser.parse_args()

    if args.use_top_ranked and args.use_deep_layers:
        raise ValueError("Cannot provide both command line options '--use-top-ranked' and '--use-deep-layers'. "
                         "Specify only one of them.")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Number of neighbors
    n_neighbors = args.num_neighbors
    if n_neighbors <= 0:
        n_neighbors = None

    output_dir = args.output_dir

    # Method name for results and plots
    method_name = METHOD_NAME_MAP[args.detection_method]
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

    elif args.detection_method == 'trust':
        # Append the layer name to the method name
        method_name = '{:.5s}_{}'.format(method_name, args.layer_trust_score)
        # If `n_neighbors` is specified, append that value to the name string
        if n_neighbors is not None:
            method_name = '{}_k{:d}'.format(method_name, n_neighbors)

    elif args.detection_method == 'dknn':
        # If `n_neighbors` is specified, append that value to the name string
        if n_neighbors is not None:
            method_name = '{}_k{:d}'.format(method_name, n_neighbors)

    elif args.detection_method in ['lid', 'lid_class_cond']:
        # If `n_neighbors` is specified, append that value to the name string
        if n_neighbors is not None:
            method_name = '{}_k{:d}'.format(method_name, n_neighbors)

    # Check if the numpy data directory exists
    d = os.path.join(NUMPY_DATA_PATH, args.model_type)
    if not os.path.isdir(d):
        raise ValueError("Directory for the numpy data files not found: {}".format(d))

    # Load the saved detection scores and labels from the test folds for the given method
    scores_folds, labels_folds, _, _ = load_detector_checkpoint(output_dir, method_name, False)

    num_folds = len(scores_folds)
    # Perturbation norm for the test folds
    norm_type = ATTACK_NORM_MAP[args.adv_attack]
    norm_folds = []
    for i in range(num_folds):
        # Load the saved clean numpy data from this fold
        numpy_save_path = get_clean_data_path(args.model_type, i + 1)
        # Temporary hack to use backup data directory
        numpy_save_path = numpy_save_path.replace('varun', 'jayaram', 1)

        data_tr, labels_tr, data_te, labels_te = load_numpy_data(numpy_save_path)
        num_clean_tr = labels_tr.shape[0]
        num_clean_te = labels_te.shape[0]

        # Load the saved adversarial numpy data generated from this training and test fold.
        # `labels_te_adv` corresponds to the class labels of the clean samples, not that predicted by the DNN
        data_tr_clean, data_te_clean, data_tr_adv, labels_tr_adv, data_te_adv, labels_te_adv = \
            load_adversarial_wrapper(i, args.model_type, args.adv_attack, args.max_attack_prop, num_clean_te)

        num_adv_tr = labels_tr_adv.shape[0]
        num_adv_te = labels_te_adv.shape[0]
        print("\nTrain fold {:d}: number of clean samples = {:d}, number of adversarial samples = {:d}, % of "
              "adversarial samples = {:.4f}".format(i + 1, num_clean_tr, num_adv_tr,
                                                    (100. * num_adv_tr) / (num_clean_tr + num_adv_tr)))
        print("Test fold {:d}: number of clean samples = {:d}, number of adversarial samples = {:d}, % of adversarial "
              "samples = {:.4f}".format(i + 1, num_clean_te, num_adv_te,
                                        (100. * num_adv_te) / (num_clean_te + num_adv_te)))

        assert data_te_clean.shape[0] == num_adv_te
        assert data_te_adv.shape[0] == num_adv_te
        # perturbation norm of test fold adversarial samples
        diff = data_te_adv.reshape(num_adv_te, -1) - data_te_clean.reshape(num_adv_te, -1)
        if norm_type == 'inf':
            norm_diff_te = np.linalg.norm(diff, ord=np.inf, axis=1)
        else:
            # expecting a non-negative integer
            norm_diff_te = np.linalg.norm(diff, ord=int(norm_type), axis=1)

        # Filling in zeros for the perturbation norm of clean test fold samples
        norm_folds.append(np.concatenate([np.zeros(num_clean_te), norm_diff_te]))

    if args.x_var == 'proportion':
        print("\nCalculating performance metrics for different proportion of attack samples:")
        # fname = os.path.join(output_dir, 'detection_metrics_{}.pkl'.format(method_name))
        fname = None
        results_dict = metrics_varying_positive_class_proportion(
            scores_folds, labels_folds, output_file=fname, max_pos_proportion=args.max_attack_prop, log_scale=False
        )
    else:
        pass

    # print("Performance metrics calculated and saved to the file: {}".format(fname))


if __name__ == '__main__':
    main()
