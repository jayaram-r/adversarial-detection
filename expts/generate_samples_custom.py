"""
Main script for generating adversarial data (from custom KNN attack) from the cross-validation folds and saving
them to numpy files.

Example usage:
python generate_samples_custom.py -m mnist --gpu 3 --defense-method dknn --dist-metric euclidean --n-jobs 16

"""
from __future__ import absolute_import, division, print_function
import sys
import argparse
import os
import pickle
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from nets.mnist import *
from nets.cifar10 import *
from nets.svhn import *
from nets.resnet import *
from helpers.constants import (
    ROOT,
    SEED_DEFAULT,
    CROSS_VAL_SIZE,
    NORMALIZE_IMAGES,
    BATCH_SIZE_DEF,
    NEIGHBORHOOD_CONST,
    CUSTOM_ATTACK
)
from helpers.utils import (
    load_model_checkpoint,
    convert_to_loader,
    load_numpy_data,
    get_data_bounds,
    verify_data_loader,
    get_samples_as_ndarray,
    get_predicted_classes
)
from helpers import knn_attack
from detectors.detector_proposed import extract_layer_embeddings as extract_layer_embeddings_numpy


def helper_accuracy(layer_embeddings, labels_pred_dnn, labels, model_detec_propo, model_detec_dknn):
    # Accuracy of the DNN classifier
    n_test = labels.shape[0]
    mask = labels_pred_dnn == labels
    accu_dnn = (100. * mask[mask].shape[0]) / n_test
    # Accuracy of the proposed method
    is_error, _ = knn_attack.check_adv_detec(layer_embeddings, labels, labels_pred_dnn, model_detec_propo,
                                             is_numpy=True)
    accu_propo = (100. * (n_test - is_error[is_error].shape[0])) / n_test
    # Accuracy of deep KNN
    is_error, _ = knn_attack.check_adv_detec(layer_embeddings, labels, labels_pred_dnn, model_detec_dknn,
                                             is_numpy=True)
    accu_dknn = (100. * (n_test - is_error[is_error].shape[0])) / n_test

    return accu_dnn, accu_propo, accu_dknn


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-batch-size', '--tb', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--output-dir', '-o', default='', help='directory path for saving the output and model files')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
    parser.add_argument('--model-type', '-m', choices=['mnist', 'cifar10', 'svhn'], default='mnist',
                        help='model type or name of the dataset')
    parser.add_argument('--seed', '-s', type=int, default=SEED_DEFAULT, help='seed for random number generation')
    parser.add_argument('--generate-attacks', type=bool, default=True,
                        help='should attack samples be generated/not (default:True)')
    parser.add_argument('--gpu', type=str, default="3", help='which gpus to execute code on')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size of evaluation')
    parser.add_argument('--defense-method', '--dm', choices=['dknn', 'proposed', 'dnn'], default='dknn',
                        help="Defense method to attack. Choices are 'dnn', 'dknn' and 'proposed'")
    parser.add_argument('--det-model-file', '--dmf', default='',
                        help='Path to the saved detector model file. Loads from a default location of not specified.')
    parser.add_argument('--dist-metric', choices=['euclidean', 'cosine'], default='euclidean',
                        help='distance metric to use')
    parser.add_argument('--n-jobs', type=int, default=16, help='number of parallel jobs to use for multiprocessing')
    parser.add_argument('--skip-subsampling', action='store_true', default=False,
                        help='Use this option to skip random sub-sampling of the train data split')
    parser.add_argument('--untargeted', action='store_true', default=False,
                        help='Use this option to create untargeted adversarial samples from this attack')
    '''
    parser.add_argument('--stepsize', type=float, default=0.001, help='stepsize')
    parser.add_argument('--max-iterations', type=int, default=1000, help='max num. of iterations')
    '''
    parser.add_argument('--num-folds', '--nf', type=int, default=CROSS_VAL_SIZE,
                        help='number of cross-validation folds')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Output directory
    if not args.output_dir:
        output_dir = os.path.join(ROOT, 'numpy_data', args.model_type)
    else:
        output_dir = args.output_dir

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    data_path = os.path.join(ROOT, 'data')
    if args.model_type == 'mnist':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(*NORMALIZE_IMAGES['mnist'])]
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train=False, download=True, transform=transform),
            batch_size=args.test_batch_size, shuffle=False, **kwargs
        )
        model = MNIST().to(device)
        model = load_model_checkpoint(model, args.model_type)
        num_classes = 10

    elif args.model_type == 'cifar10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(*NORMALIZE_IMAGES['cifar10'])]
        )
        testset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        num_classes = 10
        model = ResNet34().to(device)
        model = load_model_checkpoint(model, args.model_type)

    elif args.model_type == 'svhn':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(*NORMALIZE_IMAGES['svhn'])]
        )
        testset = datasets.SVHN(root=data_path, split='test', download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        num_classes = 10
        model = SVHN().to(device)
        model = load_model_checkpoint(model, args.model_type)

    else:
        raise ValueError("'{}' is not a valid model type".format(args.model_type))

    # Set model to evaluation mode
    model.eval()

    # convert the test data loader to 2 ndarrays
    data, labels = get_samples_as_ndarray(test_loader)

    # Get the range of values in the data array
    bounds = get_data_bounds(data)
    print("Range of data values: ({:.4f}, {:.4f})\n".format(*bounds))

    # verify if the data loader is the same as the ndarrays it generates
    if not verify_data_loader(test_loader, batch_size=args.test_batch_size):
        raise ValueError("Data loader verification failed")

    # Path to the detection model file
    det_model_file = ''
    if args.det_model_file:
        det_model_file = args.det_model_file
    else:
        if args.defense_method != 'dnn':
            # default path the the saved detection model file
            det_model_file = os.path.join(ROOT, 'outputs', args.model_type, 'detection', CUSTOM_ATTACK,
                                          'models_{}.pkl'.format(args.defense_method))

    print("Defense method: {}".format(args.defense_method))
    if det_model_file:
        print("Loading saved detection models from the file: {}".format(det_model_file))
        # Load the detection models (from each cross-validation fold) from a pickle file.
        # `models_detec` will be a list of trained detection models from each fold
        with open(det_model_file, 'rb') as fp:
            models_detec = pickle.load(fp)
    else:
        models_detec = [None] * args.num_folds

    # Detection models for the dknn method. Used for comparison
    fname = os.path.join(ROOT, 'outputs', args.model_type, 'detection', CUSTOM_ATTACK, 'models_dknn.pkl')
    with open(fname, 'rb') as fp:
        models_detec_dknn = pickle.load(fp)

    # Detection models for the proposed method. Used for comparison
    fname = os.path.join(ROOT, 'outputs', args.model_type, 'detection', CUSTOM_ATTACK, 'models_proposed.pkl')
    with open(fname, 'rb') as fp:
        models_detec_propo = pickle.load(fp)

    # repeat for each fold in the cross-validation split
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    i = 1
    for ind_tr, ind_te in skf.split(data, labels):
        data_tr = data[ind_tr, :]
        labels_tr = labels[ind_tr]
        data_te = data[ind_te, :]
        labels_te = labels[ind_te]

        # Set number of nearest neighbors based on the data size and the neighborhood constant
        n_neighbors = int(np.ceil(labels_tr.shape[0] ** NEIGHBORHOOD_CONST))
        print("\nProcessing fold {:d}".format(i))
        print("Number of nearest neighbors = {:d}".format(n_neighbors))
        
        # make dir based on fold to save data
        numpy_save_path = os.path.join(output_dir, "fold_" + str(i))
        if not os.path.isdir(numpy_save_path):
            os.makedirs(numpy_save_path)

        # save train fold to numpy_save_path or load if it exists already
        if not os.path.isfile(os.path.join(numpy_save_path, 'data_tr.npy')):
            np.save(os.path.join(numpy_save_path, 'data_tr.npy'), data_tr)
        else:
            data_tr = np.load(os.path.join(numpy_save_path, "data_tr.npy"))

        if not os.path.isfile(os.path.join(numpy_save_path, 'labels_tr.npy')):
            np.save(os.path.join(numpy_save_path, 'labels_tr.npy'), labels_tr)
        else:
            labels_tr = np.load(os.path.join(numpy_save_path, "labels_tr.npy"))
        
        # save test fold to numpy_save_path or load if it exists already
        if not os.path.isfile(os.path.join(numpy_save_path, 'data_te.npy')):
            np.save(os.path.join(numpy_save_path, 'data_te.npy'), data_te)
        else:
            data_te = np.load(os.path.join(numpy_save_path, "data_te.npy"))

        if not os.path.isfile(os.path.join(numpy_save_path, 'labels_te.npy')):
            np.save(os.path.join(numpy_save_path, 'labels_te.npy'), labels_te)
        else:
            labels_te = np.load(os.path.join(numpy_save_path, "labels_te.npy"))

        if args.generate_attacks:
            # print(data_tr.shape, labels_tr.shape)
            adv_save_path = os.path.join(output_dir, 'fold_{}'.format(i), CUSTOM_ATTACK)
            if not os.path.isdir(adv_save_path):
                os.makedirs(adv_save_path)

            n_test = labels_te.shape[0]
            n_train = labels_tr.shape[0]
            if not args.skip_subsampling:
                # Select a random, class-stratified sample from the training data of size `n_test`.
                # This is done to speed-up the attack optimization
                sss = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=args.seed)
                _, ind_sample = next(sss.split(data_tr, labels_tr))
                data_tr_sample = data_tr[ind_sample, :]
                labels_tr_sample = labels_tr[ind_sample]
                print("\nRandomly sampling the train split from {:d} to {:d} samples".format(n_train, n_test))
            else:
                data_tr_sample = data_tr
                labels_tr_sample = labels_tr

            # Data loader for the train and test split
            train_fold_loader = convert_to_loader(data_tr_sample, labels_tr_sample, batch_size=args.batch_size,
                                                  custom=False)
            test_fold_loader = convert_to_loader(data_te, labels_te, batch_size=args.batch_size, custom=False)
            # Extract the layer embeddings for samples from the train and test split
            layer_embeddings_train, _, _, _ = extract_layer_embeddings_numpy(model, device, train_fold_loader,
                                                                             method='proposed')
            layer_embeddings_test, _, labels_pred_dnn_test, _ = extract_layer_embeddings_numpy(
                model, device, test_fold_loader, method='proposed'
            )
            # Calculate accuracy of the DNN and the detection methods on clean data
            accu_clean_dnn, accu_clean_propo, accu_clean_dknn = helper_accuracy(
                layer_embeddings_test, labels_pred_dnn_test, labels_te, models_detec_propo[i - 1],
                models_detec_dknn[i - 1]
            )
            print("Accuracy on clean data:\nDNN classifier: {:.4f}, proposed: {:.4f}, dknn: {:.4f}".
                  format(accu_clean_dnn, accu_clean_propo, accu_clean_dknn))

            # Load kernel sigma values from file if available
            sigma_filename = os.path.join(adv_save_path, 'kernel_sigma_{}.npy'.format(args.dist_metric))
            if os.path.isfile(sigma_filename):
                sigma_per_layer = np.load(sigma_filename)
            else:
                # Search for suitable kernel scale per layer.
                # `sigma_per_layer` should be a numpy array of size `(data_te.shape[0], n_layers)`
                print("Setting the kernel scale values for the test fold data.")
                sigma_per_layer = knn_attack.set_kernel_scale(
                    layer_embeddings_train, layer_embeddings_test,
                    metric=args.dist_metric, n_neighbors=n_neighbors, n_jobs=args.n_jobs
                )
                np.save(sigma_filename, sigma_per_layer)

            del test_fold_loader, layer_embeddings_train, layer_embeddings_test
            # numpy array to torch tensor
            sigma_per_layer = torch.from_numpy(sigma_per_layer).to(device)
            # Index of samples from each class in `labels_tr_sample`
            labels_uniq = np.unique(labels_tr_sample)
            indices_per_class = {c: np.where(labels_tr_sample == c)[0] for c in labels_uniq}

            # `layer_embeddings_per_class_train` contains the layer wise embeddings corresponding to each class
            # from the `train_fold_loader`. It is a dict mapping each class to a list of torch tensors per layer
            layer_embeddings_per_class_train = knn_attack.extract_layer_embeddings(
                model, device, train_fold_loader, indices_per_class, split_by_class=True
            )
            print("Creating adversarial samples from the test fold.")
            # Recreating the test fold loader with `custom = True` in order to get the sample indices
            test_fold_loader = convert_to_loader(data_te, labels_te, batch_size=args.batch_size, custom=True)
            data_adver = []
            labels_adver = []
            data_clean = []
            labels_clean = []
            norm_perturb = []
            is_correct = []
            is_adver = []
            for batch_idx, (data_temp, labels_temp, index_temp) in enumerate(test_fold_loader):
                index_temp = index_temp.cpu().numpy()
                # data_batch_excl = np.delete(data_te, index_temp, axis=0)
                # labels_batch_excl = np.delete(labels_te, index_temp, axis=0)
                # main attack function
                labels_pred_temp = labels_pred_dnn_test[index_temp]
                data_adver_batch, labels_adver_batch, norm_perturb_batch, is_correct_batch, is_adver_batch = \
                    knn_attack.attack(
                        model, device, data_temp.to(device), labels_temp, labels_pred_temp,
                        layer_embeddings_per_class_train, labels_uniq, sigma_per_layer[index_temp, :],
                        model_detector=models_detec[i - 1], untargeted=args.untargeted,
                        dist_metric=args.dist_metric, fast_mode=True, verbose=True
                )
                # all returned outputs are numpy arrays
                # accumulate results from this batch
                data_adver.append(data_adver_batch)
                labels_adver.append(labels_adver_batch)
                data_clean.append(data_temp.detach().cpu().numpy())
                labels_clean.append(labels_temp.detach().cpu().numpy())
                norm_perturb.append(norm_perturb_batch)
                is_correct.append(is_correct_batch)
                is_adver.append(is_adver_batch)

            data_adver = np.concatenate(data_adver, axis=0)
            labels_adver = np.asarray(np.concatenate(labels_adver), dtype=labels_te.dtype)
            data_clean = np.concatenate(data_clean, axis=0)
            labels_clean = np.asarray(np.concatenate(labels_clean), dtype=labels_te.dtype)
            norm_perturb = np.concatenate(norm_perturb)
            is_correct = np.concatenate(is_correct)
            is_adver = np.concatenate(is_adver)

            # Calculate accuracy of the DNN and the detection methods on adversarial inputs
            data_loader = convert_to_loader(data_adver, labels_clean, batch_size=args.batch_size)
            layer_embeddings, _, labels_pred_dnn, _ = extract_layer_embeddings_numpy(
                model, device, data_loader, method='proposed'
            )
            accu_dnn, accu_propo, accu_dknn = helper_accuracy(
                layer_embeddings, labels_pred_dnn, labels_clean, models_detec_propo[i - 1], models_detec_dknn[i - 1]
            )

            n_adver = is_adver[is_adver].shape[0]
            print("\nTest fold {:d}: #samples = {:d}, #adversarial samples = {:d}, avg. perturbation norm = {:.6f}".
                  format(i, n_test, n_adver, np.mean(norm_perturb[is_adver])))
            print("Accuracy on clean and adversarial data from test fold {:d}:".format(i))
            print("method\t{}\t{}".format('accu. clean', 'accu. adver'))
            print("{}\t{:.4f}\t{:.4f}".format('DNN', accu_clean_dnn, accu_dnn))
            print("{}\t{:.4f}\t{:.4f}".format('proposed', accu_clean_propo, accu_propo))
            print("{}\t{:.4f}\t{:.4f}".format('dknn', accu_clean_dknn, accu_dknn))

            # save data to numpy files
            np.save(os.path.join(adv_save_path, 'data_te_adv.npy'), data_adver)
            np.save(os.path.join(adv_save_path, 'labels_te_adv.npy'), labels_adver)
            np.save(os.path.join(adv_save_path, 'data_te_clean.npy'), data_clean)
            np.save(os.path.join(adv_save_path, 'labels_te_clean.npy'), labels_clean)
            np.save(os.path.join(adv_save_path, 'norm_perturb.npy'), norm_perturb)
            # np.save(os.path.join(adv_save_path, 'is_correct_detec.npy'), is_correct)
            np.save(os.path.join(adv_save_path, 'is_adver.npy'), is_adver)
        else:
            print("generated original data split for fold : ", i)

        i = i + 1


if __name__ == '__main__':
    main()
