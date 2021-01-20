"""
Extract the layer representations of a trained DNN and plot the test statistics of intermediate layers.
"""
from __future__ import absolute_import, division, print_function
import sys
import argparse
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from nets.mnist import *
from nets.svhn import *
from nets.resnet import *
from helpers.constants import *
from helpers.utils import (
    load_model_checkpoint,
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
    helper_layer_embeddings,
    load_adversarial_wrapper
)
from helpers.dimension_reduction_methods import load_dimension_reduction_models
from detectors.detector_proposed import DetectorLayerStatistics


def gather_test_stats(args):
    detection_method = 'proposed'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs_loader = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Number of neighbors
    n_neighbors = args.num_neighbors
    if n_neighbors <= 0:
        n_neighbors = None

    # Model file for dimension reduction
    apply_dim_reduc = True
    model_dim_reduc = None
    if apply_dim_reduc:
        if args.modelfile_dim_reduc:
            fname = args.modelfile_dim_reduc
        else:
            # Path to the dimension reduction model file
            fname = get_path_dr_models(args.model_type, detection_method, test_statistic=args.test_statistic)

        if not os.path.isfile(fname):
            raise ValueError("Model file for dimension reduction is required, but does not exist: {}".format(fname))
        else:
            # Load the dimension reduction models for each layer from the pickle file
            model_dim_reduc = load_dimension_reduction_models(fname)

    # Pre-trained DNN model corresponding to the dataset
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

    n_samples_per_class = 5000
    test_stats_pred = {'clean': [], 'adversarial': []}
    test_stats_true = {'clean': [], 'adversarial': []}
    # Select a particular data fold
    ind_fold = 0
    for i in range(ind_fold, ind_fold + 1):
        print("\nProcessing cross-validation fold {:d}:".format(i + 1))
        # Load the saved clean numpy data from this fold
        numpy_save_path = get_clean_data_path(args.model_type, i + 1)
        # Temporary hack to use backup data directory
        numpy_save_path = numpy_save_path.replace('varun', 'jayaram', 1)
        data_tr, labels_tr, data_te, labels_te = load_numpy_data(numpy_save_path)
        num_clean_tr = labels_tr.shape[0]
        num_clean_te = labels_te.shape[0]
        # Data loader for the train fold
        train_fold_loader = convert_to_loader(data_tr, labels_tr, dtype_x=torch.float, batch_size=args.batch_size,
                                              device=device)
        # Data loader for the test fold
        test_fold_loader = convert_to_loader(data_te, labels_te, dtype_x=torch.float, batch_size=args.batch_size,
                                             device=device)
        # Get the range of values in the data array
        # bounds = get_data_bounds(np.concatenate([data_tr, data_te], axis=0))
        print("\nCalculating the layer embeddings and DNN predictions for the clean train data split:")
        layer_embeddings_tr, labels_pred_tr = helper_layer_embeddings(
            model, device, train_fold_loader, detection_method, labels_tr
        )
        print("\nCalculating the layer embeddings and DNN predictions for the clean test data split:")
        layer_embeddings_te, labels_pred_te = helper_layer_embeddings(
            model, device, test_fold_loader, detection_method, labels_te
        )
        del train_fold_loader, test_fold_loader

        # Load the saved noisy (Gaussian noise) numpy data generated from this training and test fold
        numpy_save_path = get_noisy_data_path(args.model_type, i + 1)
        # Temporary hack to use backup data directory
        numpy_save_path = numpy_save_path.replace('varun', 'jayaram', 1)
        data_tr_noisy, data_te_noisy = load_noisy_data(numpy_save_path)
        # Noisy data have the same labels as the clean data
        labels_tr_noisy = labels_tr
        labels_te_noisy = labels_te
        # Check the number of noisy samples
        assert data_tr_noisy.shape[0] == num_clean_tr, ("Number of noisy samples from the train fold is different "
                                                        "from expected")
        assert data_te_noisy.shape[0] == num_clean_te, ("Number of noisy samples from the test fold is different "
                                                        "from expected")
        # Data loader for the noisy train and test fold data
        noisy_train_fold_loader = convert_to_loader(data_tr_noisy, labels_tr_noisy, dtype_x=torch.float,
                                                    batch_size=args.batch_size, device=device)
        noisy_test_fold_loader = convert_to_loader(data_te_noisy, labels_te_noisy, dtype_x=torch.float,
                                                   batch_size=args.batch_size, device=device)
        print("\nCalculating the layer embeddings and DNN predictions for the noisy train data split:")
        layer_embeddings_tr_noisy, labels_pred_tr_noisy = helper_layer_embeddings(
            model, device, noisy_train_fold_loader, detection_method, labels_tr_noisy
        )
        print("\nCalculating the layer embeddings and DNN predictions for the noisy test data split:")
        layer_embeddings_te_noisy, labels_pred_te_noisy = helper_layer_embeddings(
            model, device, noisy_test_fold_loader, detection_method, labels_te_noisy
        )
        del noisy_train_fold_loader, noisy_test_fold_loader

        # Load the saved adversarial numpy data generated from this training and test fold
        _, data_te_clean, data_tr_adv, labels_tr_adv, data_te_adv, labels_te_adv = load_adversarial_wrapper(
            i, args.model_type, args.adv_attack, args.max_attack_prop, num_clean_te, index_adv=args.index_adv
        )
        # `labels_te_adv` corresponds to the class labels of the clean samples, not that predicted by the DNN
        labels_te_clean = labels_te_adv
        num_adv_tr = labels_tr_adv.shape[0]
        num_adv_te = labels_te_adv.shape[0]
        print("\nTrain fold: number of clean samples = {:d}, number of adversarial samples = {:d}, % of adversarial "
              "samples = {:.4f}".format(num_clean_tr, num_adv_tr, (100. * num_adv_tr) / (num_clean_tr + num_adv_tr)))
        print("Test fold: number of clean samples = {:d}, number of adversarial samples = {:d}, % of adversarial "
              "samples = {:.4f}".format(num_clean_te, num_adv_te, (100. * num_adv_te) / (num_clean_te + num_adv_te)))

        # Adversarial data loader for the train fold
        adv_train_fold_loader = convert_to_loader(data_tr_adv, labels_tr_adv, dtype_x=torch.float,
                                                  batch_size=args.batch_size, device=device)
        # Adversarial data loader for the test fold
        adv_test_fold_loader = convert_to_loader(data_te_adv, labels_te_adv, dtype_x=torch.float,
                                                 batch_size=args.batch_size, device=device)
        print("\nCalculating the layer embeddings and DNN predictions for the adversarial train data split:")
        layer_embeddings_tr_adv, labels_pred_tr_adv = helper_layer_embeddings(
            model, device, adv_train_fold_loader, detection_method, labels_tr_adv
        )
        check_label_mismatch(labels_tr_adv, labels_pred_tr_adv)
        print("\nCalculating the layer embeddings and DNN predictions for the adversarial test data split:")
        layer_embeddings_te_adv, labels_pred_te_adv = helper_layer_embeddings(
            model, device, adv_test_fold_loader, detection_method, labels_te_adv
        )
        check_label_mismatch(labels_te_adv, labels_pred_te_adv)
        del adv_train_fold_loader, adv_test_fold_loader

        # Detection labels (0 denoting clean and 1 adversarial)
        labels_detec = np.concatenate([np.zeros(labels_pred_te.shape[0], dtype=np.int),
                                       np.ones(labels_pred_te_adv.shape[0], dtype=np.int)])
        # Proposed method
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
        for cat in ('clean', 'adversarial'):
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
            # Fit the detector on clean or adversarial data from the training fold
            if cat == 'clean':
                _ = det_model.fit(layer_embeddings_tr[st_ind:], labels_tr, labels_pred_tr)
            else:
                _ = det_model.fit(layer_embeddings_tr_adv[st_ind:], labels_tr_adv, labels_pred_tr_adv)

            # Test statistics from each layer conditioned on the predicted class
            for c, arr in det_model.test_stats_pred_null.items():
                if n_samples_per_class < arr.shape[0]:
                    ind_samp = np.random.permutation(arr.shape[0])[:n_samples_per_class]
                    test_stats_pred[cat].append(arr[ind_samp, :])
                else:
                    test_stats_pred[cat].append(arr)

            # Test statistics from each layer conditioned on the true class
            for c, arr in det_model.test_stats_true_null.items():
                if n_samples_per_class < arr.shape[0]:
                    ind_samp = np.random.permutation(arr.shape[0])[:n_samples_per_class]
                    test_stats_true[cat].append(arr[ind_samp, :])
                else:
                    test_stats_true[cat].append(arr)

    test_stats_pred['clean'] = np.concatenate(test_stats_pred['clean'], axis=0)
    test_stats_pred['adversarial'] = np.concatenate(test_stats_pred['adversarial'], axis=0)
    test_stats_true['clean'] = np.concatenate(test_stats_true['clean'], axis=0)
    test_stats_true['adversarial'] = np.concatenate(test_stats_true['adversarial'], axis=0)
    return test_stats_pred, test_stats_true


def bhattacharya_coeff(samples1, samples2, bin_edges):
    hist1, _ = np.histogram(samples1, bins=bin_edges, density=False)
    hist2, _ = np.histogram(samples2, bins=bin_edges, density=False)
    nhist1 = (1. / np.sum(hist1)) * hist1
    nhist2 = (1. / np.sum(hist2)) * hist2
    return np.sum(np.sqrt(np.multiply(nhist1, nhist2)))


def plot_test_stats(args, test_stats_pred, test_stats_true):
    # Output directory
    if not args.output_dir:
        base_dir = get_output_path(args.model_type)
        output_dir = os.path.join(base_dir, 'plot_layer_stats')
    else:
        output_dir = args.output_dir

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Name string for the results
    method_name = '{:.8s}_{:.5s}'.format(METHOD_NAME_MAP['proposed'], args.test_statistic)
    if args.use_top_ranked:
        method_name = '{}_top{:d}'.format(method_name, args.num_layers)
    elif args.use_deep_layers:
        method_name = '{}_last{:d}'.format(method_name, args.num_layers)

    if args.num_neighbors > 0:
        method_name = '{}_k{:d}'.format(method_name, args.num_neighbors)

    # Number of bins in the histogram
    # get_num_bins = lambda m: 2 if m < 40 else (50 if m >= 1000 else int(np.ceil(m / 20.)))
    get_num_bins = lambda m: min(40, m)

    n_layers = test_stats_pred['clean'].shape[1]
    if n_layers <= 6:
        ind_layers = np.arange(n_layers)
        n_layers_plot = n_layers
    else:
        # include only the first 3 and the last three layers in the plot
        ind_layers = np.array([0, 1, 2, n_layers - 3, n_layers - 2, n_layers - 1], dtype=np.int)
        n_layers_plot = 6

    fig, axes = plt.subplots(nrows=2, ncols=n_layers_plot, sharex=False, sharey=True)
    legend_layer = int(np.floor(np.median(np.arange(n_layers_plot))))
    density = True
    # Test statistics conditioned on the predicted class
    i = 0
    for j, l in enumerate(ind_layers):
        # Calculate suitable bin edges using the combined clean and adversarial samples.
        # Using the Freedman-Diaconis Estimator for the bin width.
        bin_edges = np.histogram_bin_edges(
            np.concatenate([test_stats_pred['clean'][:, l], test_stats_pred['adversarial'][:, l]]), bins='fd'
        )
        bc = bhattacharya_coeff(test_stats_pred['clean'][:, l], test_stats_pred['adversarial'][:, l], bin_edges)
        # Plot the individual histograms
        x = test_stats_pred['clean'][:, l]
        axes[i, j].hist(
            x, bins=bin_edges, density=density, histtype='step', color='g', alpha=0.75,
            label=r'normal, $t^{(\ell)}_{p \,|\, \hat{c}}$'
        )
        x = test_stats_pred['adversarial'][:, l]
        axes[i, j].hist(
            x, bins=bin_edges, density=density, histtype='step', color='r', alpha=0.75,
            label=r'adversarial, $t^{(\ell)}_{p \,|\, \hat{c}}$'
        )
        axes[i, j].set_title("layer {:d}".format(l), fontsize=9, fontweight='normal')
        axes[i, j].set_xscale('log')
        axes[i, j].set_xticks([])
        # axes[i, j].set_yticks([])
        axes[i, j].tick_params(axis='y', which='major', labelsize=9)
        axes[i, j].get_xaxis().set_tick_params(which='both', size=0)
        axes[i, j].get_xaxis().set_tick_params(which='both', width=0)
        # Show the Bhattacharya coefficient in a text box
        axes[i, j].text(0.05, 0.6, 'BC = {:.2f}'.format(np.round(bc, decimals=2)), transform=axes[i, j].transAxes,
                        fontsize=7, horizontalalignment='left', verticalalignment='center')
        if j == legend_layer:
            # Legend font sizes: xx-small, x-small, small, medium, large, x-large, xx-large
            axes[i, j].legend(loc='best', prop={'size': 'x-small', 'weight': 'normal'}, frameon=True)

    # Test statistics conditioned on the true class
    i = 1
    for j, l in enumerate(ind_layers):
        # Calculate suitable bin edges using the combined clean and adversarial samples.
        # Using the Freedman-Diaconis Estimator for the bin width.
        bin_edges = np.histogram_bin_edges(
            np.concatenate([test_stats_true['clean'][:, l], test_stats_true['adversarial'][:, l]]), bins='fd'
        )
        bc = bhattacharya_coeff(test_stats_true['clean'][:, l], test_stats_true['adversarial'][:, l], bin_edges)
        # Plot the individual histograms
        x = test_stats_true['clean'][:, l]
        axes[i, j].hist(
            x, bins=bin_edges, density=density, histtype='step', color='g', alpha=0.75,
            label=r'normal, $t^{(\ell)}_{s \,|\, c}$'
        )
        x = test_stats_true['adversarial'][:, l]
        axes[i, j].hist(
            x, bins=bin_edges, density=density, histtype='step', color='r', alpha=0.75,
            label=r'adversarial, $t^{(\ell)}_{s \,|\, c}$'
        )
        axes[i, j].set_xscale('log')
        axes[i, j].set_xticks([])
        # axes[i, j].set_yticks([])
        axes[i, j].tick_params(axis='y', which='major', labelsize=9)
        axes[i, j].get_xaxis().set_tick_params(which='both', size=0)
        axes[i, j].get_xaxis().set_tick_params(which='both', width=0)
        # Show the Bhattacharya coefficient in a text box
        axes[i, j].text(0.05, 0.6, 'BC = {:.2f}'.format(np.round(bc, decimals=2)), transform=axes[i, j].transAxes,
                        fontsize=7, horizontalalignment='left', verticalalignment='center')
        if j == legend_layer:
            # Legend font sizes: xx-small, x-small, small, medium, large, x-large, xx-large
            axes[i, j].legend(loc='best', prop={'size': 'x-small', 'weight': 'normal'}, frameon=True)

    # fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '{}.png'.format(method_name)), dpi=600, bbox_inches='tight',
                transparent=False)
    fig.savefig(os.path.join(output_dir, '{}.pdf'.format(method_name)), dpi=600, bbox_inches='tight',
                transparent=False)
    plt.close(fig)


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256, help='batch size of evaluation')
    parser.add_argument('--model-type', '-m', choices=['mnist', 'cifar10', 'svhn'], default='mnist',
                        help='model type or name of the dataset')
    parser.add_argument('--index-adv', type=int, default=0,
                        help='Index of the adversarial attack parameter to use. This indexes the sorted directories '
                             'containing the adversarial data files from different attack parameters.')
    # ------------------ Optional arguments for the proposed method
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
    # ------------------ Optional arguments for the proposed method
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

    return args


if __name__ == '__main__':
    args = parse_cli()
    test_stats_pred, test_stats_true = gather_test_stats(args)
    plot_test_stats(args, test_stats_pred, test_stats_true)
