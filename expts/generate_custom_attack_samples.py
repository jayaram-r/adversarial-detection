"""
Main script for generating adversarial data (from custom KNN attack) from the cross-validation folds and saving
them to numpy files.
"""
from __future__ import absolute_import, division, print_function
import sys
import argparse
import os
import pickle
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
    NORMALIZE_IMAGES,
    BATCH_SIZE_DEF,
    NEIGHBORHOOD_CONST,
    CUSTOM_ATTACK,
    MAX_NUM_REPS
)
from helpers.utils import (
    load_model_checkpoint,
    convert_to_loader,
    load_numpy_data,
    get_clean_data_path,
    get_adversarial_data_path,
    get_data_bounds,
    verify_data_loader,
    get_samples_as_ndarray
)
from helpers import knn_attack
from detectors.detector_proposed import extract_layer_embeddings


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
    parser.add_argument('--generate-attacks', type=bool, default=True, help='should attack samples be generated/not (default:True)')
    parser.add_argument('--gpu', type=str, default="3", help='which gpus to execute code on')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size of evaluation')
    parser.add_argument('--det-model-file', '--dmf',
                        default='/nobackup/varun/adversarial-detection/expts/outputs/mnist/detection/CW/deep_knn/models_deep_KNN.pkl',
                        help='Path to the saved detector model file')
    parser.add_argument('--dist-metric', choices=['euclidean', 'cosine'], default='euclidean',
                        help='distance metric to use')
    parser.add_argument('--n-jobs', type=int, default=16, help='number of parallel jobs to use for multiprocessing')
    '''
    parser.add_argument('--stepsize', type=float, default=0.001, help='stepsize')
    parser.add_argument('--max-iterations', type=int, default=1000, help='max num. of iterations')
    '''
    parser.add_argument('--num-folds', '--nf', type=int, default=CROSS_VAL_SIZE,
                        help='number of cross-validation folds')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not args.output_dir:
        output_dir = os.path.join(ROOT, 'numpy_data', args.model_type)
    else:
        output_dir = args.output_dir

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
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

    # Load the deep KNN models from a pickle file
    with open(args.det_model_file, 'rb') as fp:
        models_dknn = pickle.load(fp)

    # `models_dknn` will be a list of trained deep KNN models from each fold

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
            print(data_tr.shape, labels_tr.shape)
            adv_save_path = os.path.join(output_dir, 'fold_{}'.format(i), CUSTOM_ATTACK)
            if not os.path.isdir(adv_save_path):
                os.makedirs(adv_save_path)

            # Data loaders for the train and test split
            train_fold_loader = convert_to_loader(data_tr, labels_tr, batch_size=args.batch_size, custom=True)
            test_fold_loader = convert_to_loader(data_te, labels_te, batch_size=args.batch_size, custom=True)

            # Extract the layer embeddings for samples from the test split
            layer_embeddings_test, _, _, _ = extract_layer_embeddings(model, device, test_fold_loader, method='dknn')
            # `layer_embeddings_test` will be a list of numpy arrays

            # Create attack data for samples from the test split
            # Search for suitable kernel scale per layer.
            # `sigma_per_layer` should be a torch tensor of size `(data_te.shape[0], n_layers)`
            print("Setting the kernel scale values for the test fold data.")
            sigma_per_layer = knn_attack.set_kernel_scale(layer_embeddings_test, labels_te, metric=args.dist_metric,
                                                          n_neighbors=n_neighbors, n_jobs=args.n_jobs)
            print("Creating adversarial samples from the test fold.")
            for batch_idx, (data_temp, labels_temp, index_temp) in enumerate(test_fold_loader):
                index_temp = index_temp.cpu().numpy()
                data_batch_excl = np.delete(data_te, index_temp, axis=0)
                labels_batch_excl = np.delete(labels_te, index_temp, axis=0)
                data_loader = convert_to_loader(data_batch_excl, labels_batch_excl, batch_size=args.batch_size)

                data_adv_batch = knn_attack.attack(
                    model, device, data_loader, labels_batch_excl, data_temp.to(device), labels_temp,
                    models_dknn[i - 1], sigma_per_layer[index_temp, :], dist_metric=args.dist_metric
                )

        else:
            print("generated original data split for fold : ", i)

        i = i + 1


if __name__ == '__main__':
    main()
