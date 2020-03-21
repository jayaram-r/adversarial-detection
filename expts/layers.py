"""
Extract layer embeddings of a dataset given a trained DNN, estimate the intrinsic dimensionality of each layer
embedding, and search for the best projected (reduced) dimension using a dimensionality reduction method such as
neighborhood preserving projection (NPP) or PCA. The dimensionality reduction model is saved to a pickle file that
can be loaded and applied to new (test) data samples.

"""
from __future__ import absolute_import, division, print_function
import argparse
import torch
from torchvision import datasets, transforms
from nets.mnist import *
from nets.cifar10 import *
from nets.resnet import *
from nets.svhn import *
import os
import foolbox
import sys
from pympler.asizeof import asizeof
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from helpers.knn_classifier import knn_parameter_search
from helpers.lid_estimators import estimate_intrinsic_dimension
from helpers.dimension_reduction_methods import (
    transform_data_from_model,
    wrapper_data_projection
)
from helpers.constants import (
    ROOT,
    NEIGHBORHOOD_CONST,
    METRIC_DEF,
    PCA_CUTOFF,
    METHOD_INTRINSIC_DIM,
    METHOD_DIM_REDUCTION,
    NORMALIZE_IMAGES,
    MAX_SAMPLES_DIM_REDUCTION,
    DETECTION_METHODS
)
from helpers.utils import load_model_checkpoint, get_num_jobs
from detectors.detector_proposed import extract_layer_embeddings
try:
    import cPickle as pickle
except:
    import pickle


def search_dimension_and_neighbors(embeddings, labels, embeddings_test, labels_test, indices_sample, model_file,
                                   output_file, n_jobs):
    num_k_values = 10
    num_dim_values = 20
    dim_min = 50
    dim_min_pca = 1000

    n_layers = len(embeddings)
    print("\nNumber of layers = {}".format(n_layers))
    # Projection model from the different layers
    model_projection_layers = []
    # Projected (dimension reduced) training and test data from the different layers
    # data_train_layers = []
    # data_test_layers = []
    for i in range(n_layers):
        lines = []
        mode = 'w' if i == 0 else 'a'
        output_fp = open(output_file, mode)
        str0 = "\nLayer: {:d}".format(i + 1)
        print(str0)
        lines.append(str0 + '\n')

        # Embeddings from layer i
        data = embeddings[i]
        data_test = embeddings_test[i]
        if (labels.shape[0] != data.shape[0]) or (labels_test.shape[0] != data_test.shape[0]):
            raise ValueError("Mismatch in the size of the data and labels array!")

        # Random stratified sample from the training portion of the data
        data_sample = data[indices_sample, :]
        labels_sample = labels[indices_sample]
        dim_orig = data_sample.shape[1]
        n_samples = labels_sample.shape[0]
        str0 = ("Original dimension = {:d}. Train data size = {:d}. Test data size = {:d}. Sub-sample size used "
                "for dimension reduction = {:d}".format(dim_orig, labels.shape[0], labels_test.shape[0],
                                                        n_samples))
        print(str0)
        lines.append(str0 + '\n')

        d = estimate_intrinsic_dimension(data_sample, method=METHOD_INTRINSIC_DIM, n_jobs=n_jobs)
        d = min(int(np.ceil(d)), dim_orig)
        str0 = "Intrinsic dimensionality: {:d}".format(d)
        print(str0)
        lines.append(str0 + '\n')

        # Search values for the number of neighbors `k`
        k_max = int(n_samples ** NEIGHBORHOOD_CONST)
        k_range = np.unique(np.linspace(1, k_max, num=num_k_values, dtype=np.int))
        if dim_orig > dim_min:
            print("\nSearching for the best number of neighbors (k) and projected dimension.")
            d_max = min(10 * d, dim_orig)
            dim_proj_range = np.unique(np.linspace(d, d_max, num=num_dim_values, dtype=np.int))
            # Apply PCA pre-processing prior to NPP only if the data dimension exceeds 1000
            pc = 1.0 if (dim_orig < dim_min_pca) else PCA_CUTOFF

            k_best, dim_best, error_rate_cv, _, model_projection = knn_parameter_search(
                data_sample, labels_sample, k_range,
                dim_proj_range=dim_proj_range,
                method_proj=METHOD_DIM_REDUCTION,
                metric=METRIC_DEF,
                pca_cutoff=pc,
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
        output_fp.writelines(lines)
        output_fp.close()

        # print("\nProjecting the entire train and test data to {:d} dimensions:".format(dim_best))
        # data_train_layers.append(transform_data_from_model(data, model_projection))
        # data_test_layers.append(transform_data_from_model(data_test, model_projection))

    with open(model_file, 'wb') as fp:
        pickle.dump(model_projection_layers, fp)

    print("\nOutputs saved to the file: {}".format(output_file))
    print("Dimension reduction models saved to the file: {}".format(model_file))


def project_fixed_dimension(embeddings, labels, embeddings_test, labels_test, dim_proj, indices_sample,
                            model_file, output_file, n_jobs):
    n_layers = len(embeddings)
    print("\nNumber of layers = {}".format(n_layers))
    # Projection model from the different layers
    model_projection_layers = []
    lines = []
    for i in range(n_layers):
        str0 = "\nLayer: {:d}".format(i + 1)
        print(str0)
        lines.append(str0 + '\n')

        # Embeddings from layer i
        data = embeddings[i]
        data_test = embeddings_test[i]
        if (labels.shape[0] != data.shape[0]) or (labels_test.shape[0] != data_test.shape[0]):
            raise ValueError("Mismatch in the size of the data and labels array!")

        dim_orig = data.shape[1]
        if dim_orig <= dim_proj:
            str0 = "Original dimension = {:d}. Skipping dimensionality reduction for this layer".format(dim_orig)
            print(str0)
            lines.append(str0 + '\n')
            model_projection = None
        else:
            # Random stratified sample from the training portion of the data
            data_sample = data[indices_sample, :]
            labels_sample = labels[indices_sample]
            str0 = ("Original dimension = {:d}. Train data size = {:d}. Test data size = {:d}. Sub-sample size used "
                    "for dimension reduction = {:d}".format(dim_orig, labels.shape[0], labels_test.shape[0],
                                                            labels_sample.shape[0]))
            print(str0)
            lines.append(str0 + '\n')
            model_projection, _ = wrapper_data_projection(
                data_sample,
                METHOD_DIM_REDUCTION,
                dim_proj=dim_proj,
                metric=METRIC_DEF,
                pca_cutoff=PCA_CUTOFF,
                n_jobs=n_jobs
            )

        model_projection_layers.append(model_projection)

    with open(output_file, 'w') as fp:
        fp.writelines(lines)
    with open(model_file, 'wb') as fp:
        pickle.dump(model_projection_layers, fp)

    print("\nOutputs saved to the file: {}".format(output_file))
    print("Dimension reduction models saved to the file: {}".format(model_file))


def main():
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--model-type', '-m', choices=['mnist', 'cifar10', 'svhn'], default='mnist',
                        help='model type or name of the dataset')
    parser.add_argument('--detection-method', '--dm', choices=DETECTION_METHODS, default='proposed',
                        help="Detection method to run. Choices are: {}".format(', '.join(DETECTION_METHODS)))
    parser.add_argument('--fixed-dimension', '--fd', type=int, default=0,
                        help='Use this option to project the layer embeddings to a fixed dimension, if a layer '
                             'dimension exceeds this value. Zero or a negative value disables this option.')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', '--tb', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', '-s', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--n-jobs', type=int, default=8, help='number of parallel jobs to use for multiprocessing')
    parser.add_argument('--gpu', type=str, default='2', help='gpus to execute code on')
    parser.add_argument('--output-dir', '-o', type=str, default='', help='output directory path')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    n_jobs = get_num_jobs(args.n_jobs)

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(ROOT, 'outputs', args.model_type)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

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
            datasets.MNIST(data_path, train=False, download=True, transform=transform),
            batch_size=args.test_batch_size, shuffle=True, **kwargs
        )
        model = MNIST().to(device)
        num_classes = 10

    elif args.model_type == 'cifar10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(*NORMALIZE_IMAGES['cifar10'])]
        )
        trainset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        testset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        model = ResNet34().to(device)
        num_classes = 10
    
    elif args.model_type == 'svhn':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(*NORMALIZE_IMAGES['svhn'])]
        )
        trainset = datasets.SVHN(root=data_path, split='train', download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        testset = datasets.SVHN(root=data_path, split='test', download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        model = SVHN().to(device)
        num_classes = 10
    
    else:
        raise ValueError("'{}' is not a valid model type".format(args.model_type))

    # Load the saved model checkpoint and set it to eval mode
    model = load_model_checkpoint(model, args.model_type)
    model.eval()

    # Get the feature embeddings from all the layers and the labels
    print("Calculating layer embeddings for the train data:")
    embeddings, labels, labels_pred, counts = extract_layer_embeddings(
        model, device, train_loader, method=args.detection_method
    )
    print("\nCalculating layer embeddings for the test data:")
    embeddings_test, labels_test, labels_pred_test, counts_test = extract_layer_embeddings(
        model, device, test_loader, method=args.detection_method
    )
    accu_test = np.sum(labels_test == labels_pred_test) / float(labels_test.shape[0])
    print("\nTest set accuracy = {:.4f}".format(accu_test))

    ns = labels.shape[0]
    if ns > MAX_SAMPLES_DIM_REDUCTION:
        # Take a random class-stratified subsample of the data for intrinsic dimension estimation and
        # dimensionality reduction
        sss = StratifiedShuffleSplit(n_splits=1, test_size=MAX_SAMPLES_DIM_REDUCTION, random_state=args.seed)
        temp = np.zeros((ns, 2))   # placeholder data array
        _, indices_sample = next(sss.split(temp, labels))
    else:
        indices_sample = np.arange(ns)

    if args.fixed_dimension < 1:
        output_file = os.path.join(output_dir, 'output_layer_extraction.txt')
        model_file = os.path.join(output_dir, 'models_dimension_reduction.pkl')
        # Search for the best number of dimensions and number of neighbors and save the corresponding projection model
        search_dimension_and_neighbors(embeddings, labels, embeddings_test, labels_test, indices_sample, model_file,
                                       output_file, n_jobs)
    else:
        if args.detection_method in ['lid', 'lid_class_cond']:
            # This method uses a different (larger) number of layer embeddings
            output_file = os.path.join(output_dir, "output_fixed_dimension_{:d}_lid.txt".format(args.fixed_dimension))
            model_file = os.path.join(output_dir, "models_fixed_dimension_{:d}_lid.pkl".format(args.fixed_dimension))
        else:
            output_file = os.path.join(output_dir, "output_fixed_dimension_{:d}.txt".format(args.fixed_dimension))
            model_file = os.path.join(output_dir, "models_fixed_dimension_{:d}.pkl".format(args.fixed_dimension))

        # Project the embeddings from each layer to the specified fixed dimension, if it exceeds the fixed dimension
        project_fixed_dimension(embeddings, labels, embeddings_test, labels_test, args.fixed_dimension,
                                indices_sample, model_file, output_file, n_jobs)


if __name__ == '__main__':
    main()
