"""
Main script for generating noisy data from the cross-validation folds and saving them to numpy files.

Example usage:
# This searches for the best noise standard deviation and generates the noisy data files
python generate_noisy_data.py -m mnist

# This uses the value 0.25 as the upper bound of the noise standard deviation
python generate_noisy_data.py -m mnist --stdev-high 0.25

"""
from __future__ import absolute_import, division, print_function
import sys
import argparse
import os
import shutil
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
    NUM_NOISE_VALUES,
    BATCH_SIZE_DEF
)
from helpers.utils import (
    load_model_checkpoint,
    load_numpy_data,
    get_clean_data_path,
    verify_data_loader,
    get_samples_as_ndarray
)
from helpers.noisy import (
    get_noise_stdev,
    create_noisy_samples
)


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--stdev-high', type=float, default=-1.0,
                        help="Upper bound on the noise standard deviation. Use the option '--search-noise-stdev' "
                             "to set this value automatically")
    parser.add_argument('--stdev-low', type=float, default=-1.0, help="Lower bound on the noise standard deviation")
    parser.add_argument('--test-batch-size', '--tb', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--output-dir', '-o', default='', help='directory path for saving the output and model files')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--model-type', '-m', choices=['mnist', 'cifar10', 'svhn'], default='cifar10',
                        help='model type or name of the dataset')
    parser.add_argument('--seed', '-s', type=int, default=SEED_DEFAULT, help='seed for random number generation')
    parser.add_argument('--gpu', type=str, default='2', help='which gpus to execute code on')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE_DEF, help='batch size of evaluation')
    parser.add_argument('--search-noise-stdev', '--sns', action='store_true', default=False,
                        help='use option to search for a suitable noise standard deviation')
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

    # convert the test data loader to 2 ndarrays
    data, labels = get_samples_as_ndarray(test_loader)

    # verify if the data loader is the same as the ndarrays it generates
    if not verify_data_loader(test_loader, batch_size=args.test_batch_size):
        raise ValueError("Data loader verification failed")

    stdev_high = args.stdev_high
    if args.search_noise_stdev or (stdev_high < 0.):
        # Search for a suitable noise standard deviation
        stdev_high = get_noise_stdev(model, device, data, labels, seed=args.seed)
        if stdev_high is None:
            print("\nERROR: no good noise standard deviation found. Try searching over a larger range of values.")
            return

    # Noise standard deviation values
    stdev_low = args.stdev_low
    if (stdev_low < 0.) or (stdev_low >= stdev_high):
        stdev_low = stdev_high / 16.

    stdev_values = np.linspace(stdev_low, stdev_high, num=NUM_NOISE_VALUES)

    # repeat for each fold in the cross-validation split
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    # Indicates whether the train and test data from a fold was loaded from file
    loaded_from_file = np.zeros(args.num_folds, dtype=np.bool)
    i = 1
    for ind_tr, ind_te in skf.split(data, labels):
        data_tr = data[ind_tr, :]
        labels_tr = labels[ind_tr]
        data_te = data[ind_te, :]
        labels_te = labels[ind_te]

        numpy_save_path = os.path.join(output_dir, "fold_{}".format(i))
        if not os.path.isdir(numpy_save_path):
            # Create directory for this fold and save the data to numpy files
            os.makedirs(numpy_save_path)
            np.save(os.path.join(numpy_save_path, 'data_tr.npy'), data_tr)
            np.save(os.path.join(numpy_save_path, 'labels_tr.npy'), labels_tr)
            np.save(os.path.join(numpy_save_path, 'data_te.npy'), data_te)
            np.save(os.path.join(numpy_save_path, 'labels_te.npy'), labels_te)
            loaded_from_file[i - 1] = False
        else:
            # load existing data files
            data_tr, labels_tr, data_te, labels_te = load_numpy_data(numpy_save_path)
            loaded_from_file[i - 1] = True

        # Directory for noisy train and test data from this fold
        noise_base_path = os.path.join(output_dir, 'fold_{}'.format(i), 'noise_gaussian')
        if os.path.isdir(noise_base_path):
            # Clear out any old data files
            shutil.rmtree(noise_base_path)

        os.makedirs(noise_base_path)
        # Generate noisy data from the train and test fold for different standard deviation values and save them
        # to numpy files
        filenames_train = []
        filenames_test = []
        for sig in stdev_values:
            noise = np.random.normal(loc=0., scale=sig, size=data_tr.shape)
            data_tr_noisy = data_tr + noise
            noise = np.random.normal(loc=0., scale=sig, size=data_te.shape)
            data_te_noisy = data_te + noise

            fname = os.path.join(noise_base_path, 'data_tr_noisy_stdev_{:.6f}.npy'.format(sig))
            np.save(fname, data_tr_noisy)
            filenames_train.append(fname + '\n')

            fname = os.path.join(noise_base_path, 'data_te_noisy_stdev_{:.6f}.npy'.format(sig))
            np.save(fname, data_te_noisy)
            filenames_test.append(fname + '\n')

        print("Saved noisy data files from fold {:d}.".format(i))
        fname = os.path.join(noise_base_path, 'filenames_train.txt')
        with open(fname, 'w') as fp:
            fp.writelines(filenames_train)

        print("List of filenames for noisy train data from this fold can be found in the file: {}".format(fname))

        fname = os.path.join(noise_base_path, 'filenames_test.txt')
        with open(fname, 'w') as fp:
            fp.writelines(filenames_test)

        print("List of filenames for noisy test data from this fold can be found in the file: {}".format(fname))
        print('\n')
        i = i + 1

    if not (np.all(loaded_from_file) or np.all(np.logical_not(loaded_from_file))):
        raise ValueError("Unexpected error: some of the data files from the train and test folds may not "
                         "be consistent.")


if __name__ == '__main__':
    main()
