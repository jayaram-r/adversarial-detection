from __future__ import absolute_import, division, print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from nets.mnist import *
from nets.cifar10 import *
from nets.svhn import *
import os
import foolbox
import sys
from pympler.asizeof import asizeof
import numpy as np
from multiprocessing import cpu_count
from sklearn.model_selection import StratifiedShuffleSplit
from helpers.knn_classifier import knn_parameter_search
from helpers.lid_estimators import estimate_intrinsic_dimension
from helpers.dimension_reduction_methods import (
    wrapper_data_projection,
    transform_data_from_model,
    load_dimension_reduction_models
)
from .constants import (
    ROOT,
    NEIGHBORHOOD_CONST,
    METRIC_DEF,
    PCA_CUTOFF,
    METHOD_INTRINSIC_DIM,
    METHOD_DIM_REDUCTION,
    NORMALIZE_IMAGES,
    MAX_SAMPLES_DIM_REDUCTION
)
from .utils import load_model_checkpoint
try:
    import cPickle as pickle
except:
    import pickle


def combine_and_vectorize(data_batches):
    """
    Combine a list of data batches and vectorize them if they are tensors. If there is only a single data batch,
    it can be passed in as list with a single array.

    :param data_batches: list of numpy arrays containing the data batches. Each array has shape `(n, d1, ...)`,
                         where `n` can be different across the batches, but the remaining dimensions should be
                         the same.
    :return: single numpy array with the combined, vectorized data.
    """
    data = np.concatenate(data_batches, axis=0)
    s = data.shape
    if len(s) > 2:
        data = data.reshape((s[0], -1))

    return data


def extract_layer_embeddings(model, device, data_loader, num_samples=None):
    labels = []
    labels_pred = []
    embeddings = []
    num_samples_partial = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            temp = target.detach().cpu().numpy()
            labels.extend(temp)
            num_samples_partial += temp.shape[0]
            # print(batch_idx)

            # Predicted class
            outputs = model(data)
            _, predicted = outputs.max(1)
            labels_pred.extend(predicted.detach().cpu().numpy())
            # Layer outputs
            outputs_layers = model.layer_wise(data)
            if batch_idx > 0:
                for i in range(len(outputs_layers)):    # each layer
                    embeddings[i].append(outputs_layers[i].detach().cpu().numpy())
            else:
                embeddings = [[v.detach().cpu().numpy()] for v in outputs_layers]

            if num_samples:
                if num_samples_partial >= num_samples:
                    break

    # `embeddings` will be a list of length equal to the number of layers.
    # `embeddings[i]` will be a list of numpy arrays corresponding to the data batches for layer `i`.
    # `embeddings[i][j]` will have shape `(b, d1, d2, d3)` or `(b, d1)` where `b` is the batch size and the rest
    # are dimensions.
    labels = np.array(labels, dtype=np.int)
    labels_pred = np.array(labels_pred, dtype=np.int)
    # Unique label counts
    labels_uniq, counts = np.unique(labels, return_counts=True)
    for a, b in zip(labels_uniq, counts):
        print("label {}, count = {:d}, proportion = {:.4f}".format(a, b, b / labels.shape[0]))

    if (np.max(counts) / np.min(counts)) >= 1.2:
        print("WARNING: classes are not balanced")

    return embeddings, labels, labels_pred, counts


def transform_layer_embeddings(embeddings_in, transform_models=None, transform_models_file=None):
    """
    Perform dimension reduction on the data embeddings from each layer. The transformation or projection matrix
    for each layer is provided via one of the inputs `transform_models` or `transform_models_file`. Only one of
    them should be specified. In the case of `transform_models_file`, the models are loaded from a pickle file.

    NOTE: In order to not perform dimension reduction at a particular layer, the corresponding element of
    `transform_models` can be set to `None`. Thus, a list of `None` values can be passed to completely skip
    dimension reduction.

    :param embeddings_in: list of data embeddings per layer. `embeddings_in[i]` is a list of numpy arrays
                          corresponding to the data batches from layer `i`.
    :param transform_models: None or a list of dictionaries with the transformation models per layer. The length of
                             `transform_models` should be equal to the length of `embeddings_in`.
    :param transform_models_file: None or a string with the file path. This should be a pickle file with the saved
                                  transformation models per layer.

    :return: list of transformed data arrays, one per layer.
    """
    if transform_models_file is None:
        if transform_models is None:
            raise ValueError("Both inputs 'transform_models' and 'transform_models_file' are not specified.")

    else:
        transform_models = load_dimension_reduction_models(transform_models_file)

    n_layers = len(embeddings_in)
    assert len(transform_models) == n_layers, "Length of 'transform_models' is not equal to the length of 'embeddings_in'"
    embeddings_out = []
    for i in range(n_layers):
        print("Transforming embeddings from layer {:d}".format(i + 1))
        data_in = combine_and_vectorize(embeddings_in[i])

        embeddings_out.append(transform_data_from_model(data_in, transform_models[i]))

    return embeddings_out


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--model-type', '-m', choices=['mnist', 'cifar10', 'svhn'], default='mnist',
                        help='model type or name of the dataset')
    parser.add_argument('--batch-size', '-b', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', '--tb', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', '-e', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', '-g', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', '-s', type=int, default=1, metavar='S', help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=100, metavar='N',
    #                     help='number of batches to wait before logging training status')
    # parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    # parser.add_argument('--adv-attack', '--aa', choices=['FGSM', 'PGD', 'CW'], default='FGSM',
    #                     help='type of adversarial attack')
    # parser.add_argument('--attack', action='store_true', default=False, help='option to launch adversarial attack')
    # parser.add_argument('--p-norm', '-p', choices=['2', 'inf'], default='inf',
    #                     help="p norm for the adversarial attack; options are '2' and 'inf'")
    # parser.add_argument('--train', '-t', action='store_true', default=False, help='commence training')
    # parser.add_argument('--ckpt', action='store_true', default=True, help='Use the saved model checkpoint')
    parser.add_argument('--gpu', type=str, default='2', help='gpus to execute code on')
    parser.add_argument('--output', '-o', type=str, default='output_layer_extraction.txt',
                        help='output file basename')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    data_path = os.path.join(ROOT, 'data')
    if args.model_type == 'mnist':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(*NORMALIZE_IMAGES['mnist'])]
        )
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train=True, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train=False, transform=transform),
            batch_size=args.test_batch_size, shuffle=True, **kwargs
        )
        model = MNIST().to(device)
        num_classes = 10

    elif args.model_type == 'cifar10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(*NORMALIZE_IMAGES['cifar10'])]
        )
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        model = CIFAR10().to(device)
        num_classes = 10
    
    elif args.model_type == 'svhn':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(*NORMALIZE_IMAGES['svhn'])]
        )
        trainset = torchvision.datasets.SVHN(root=data_path, split='train', download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        testset = torchvision.datasets.SVHN(root=data_path, split='test', download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        model = SVHN().to(device)
        num_classes = 10
    
    else:
        raise ValueError("'{}' is not a valid model type".format(args.model_type))

    # Load the saved model checkpoint
    model = load_model_checkpoint(model, args.model_type)

    # Get the feature embeddings from all the layers and the labels
    embeddings, labels, labels_pred, counts = extract_layer_embeddings(model, device, train_loader)
    embeddings_test, labels_test, labels_pred_test, counts_test = extract_layer_embeddings(model, device,
                                                                                           test_loader)
    print("Layer embeddings calculated!")
    accu_test = np.sum(labels_test == labels_pred_test) / float(labels_test.shape[0])
    print("Test set accuracy = {:.4f}".format(accu_test))

    ns = labels.shape[0]
    if ns > MAX_SAMPLES_DIM_REDUCTION:
        # Take a random class-stratified subsample of the data for intrinsic dimension estimation and
        # dimensionality reduction
        sss = StratifiedShuffleSplit(n_splits=1, test_size=MAX_SAMPLES_DIM_REDUCTION, random_state=args.seed)
        temp = np.zeros((ns, 2))   # placeholder data array
        _, indices_sample = next(sss.split(temp, labels))
    else:
        indices_sample = np.arange(ns)

    n_layers = len(embeddings)
    print("Number of layers = {}".format(n_layers))
    # Use half the number of available cores
    cc = cpu_count()
    n_jobs = max(1, int(0.5 * cc))

    output_dir = os.path.join(ROOT, 'outputs', args.model_type)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, args.output)
    # Projection model from the different layers
    model_projection_layers = []
    # Projected (dimension reduced) training and test data from the different layers
    # data_train_layers = []
    # data_test_layers = []
    for i in range(n_layers):    # number of layers in the CNN
        lines = []
        mode = 'w' if i == 0 else 'a'
        output_fp = open(output_file, mode)
        str0 = "\nLayer: {:d}".format(i + 1)
        print(str0)
        lines.append(str0 + '\n')

        # Embeddings from layer i
        data = combine_and_vectorize(embeddings[i])
        data_test = combine_and_vectorize(embeddings_test[i])

        if (labels.shape[0] != data.shape[0]) or (labels_test.shape[0] != data_test.shape[0]):
            raise ValueError("Mismatch in the size of the data and labels array!")

        # Random stratified sample from the training portion of the data
        data_sample = data[indices_sample, :]
        labels_sample = labels[indices_sample]

        dim_orig = data_sample.shape[1]
        N_samples = data_sample.shape[0]
        str0 = ("Original dimension = {:d}. Train data size = {:d}. Test data size = {:d}. Sub-sample size used "
                "for dimension reduction = {:d}".format(dim_orig, labels.shape[0], labels_test.shape[0],
                                                        labels_sample.shape[0]))
        print(str0)
        lines.append(str0 + '\n')

        d = estimate_intrinsic_dimension(data_sample, method=METHOD_INTRINSIC_DIM, n_jobs=n_jobs)
        d = min(int(np.ceil(d)), dim_orig)
        str0 = "Intrinsic dimensionality: {:d}".format(d)
        print(str0)
        lines.append(str0 + '\n')

        k_max = int(N_samples ** NEIGHBORHOOD_CONST)
        k_range = np.unique(np.linspace(1, k_max, num=10, dtype=np.int))
        if dim_orig > 20:
            print("\nSearching for the best number of neighbors (k) and projected dimension.")
            d_max = min(10 * d, dim_orig)
            dim_proj_range = np.unique(np.linspace(d, d_max, num=20, dtype=np.int))

            k_best, dim_best, error_rate_cv, _, model_projection = knn_parameter_search(
                data_sample, labels_sample, k_range,
                dim_proj_range=dim_proj_range,
                method_proj=METHOD_DIM_REDUCTION,
                metric=METRIC_DEF,
                pca_cutoff=PCA_CUTOFF,
                n_jobs=n_jobs
            )
        else:
            print("\nSkipping dimensionality reduction for this layer. Searching for the best number of "
                  "neighbors (k).")
            k_best, dim_best, error_rate_cv, _, model_projection = knn_parameter_search(
                data_sample, labels_sample, k_range,
                metric=METRIC_DEF,
                skip_preprocessing=True,
                n_jobs=n_jobs
            )

        model_projection_layers.append(model_projection)
        str_list = ["k_best: {:d}".format(k_best), "dim_best: {:d}".format(dim_best),
                    "error_rate_cv = {:.6f}".format(error_rate_cv)]
        str0 = '\n'.join(str_list)
        print(str0)
        lines.append(str0 + '\n')

        # print("\nProjecting the entire train and test data to {:d} dimensions:".format(dim_best))
        # data_train_layers.append(transform_data_from_model(data, model_projection))
        # data_test_layers.append(transform_data_from_model(data_test, model_projection))

        output_fp.writelines(lines)
        output_fp.close()

    print("\nOutputs saved to the file: {}".format(output_file))
    fname = os.path.join(output_dir, 'models_dimension_reduction.pkl')
    with open(fname, 'wb') as fp:
        pickle.dump(model_projection_layers, fp)

    print("Dimension reduction models saved to the file: {}".format(fname))


if __name__ == '__main__':
    main()
