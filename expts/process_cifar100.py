# Utility to load and process the CIFAR-100 dataset.
import os
import csv
import pdb
from PIL import Image
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torchvision import datasets, transforms
from helpers.constants import ROOT, DATA_PATH, NUMPY_DATA_PATH, NORMALIZE_IMAGES, SEED_DEFAULT

USE_GPU = False
BATCH_SIZE = 128

# Labels from the CIFAR-100 dataset that are similar to the CIFAR-10 labels. Not including them for the OOD
# detection experiment
EXCLUDE_LABELS = ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train', 'lawn_mower', 'rocket', 'streetcar',
                  'tank', 'tractor', 'bear', 'leopard', 'lion', 'tiger', 'wolf', 'camel', 'cattle', 'kangaroo',
                  'fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'crab', 'lobster', 'snail', 'spider', 'crocodile',
                  'lizard', 'turtle', 'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']


def process_data(data_set, data_loader, device, write_path, cross_val, n_folds=5, suffix=''):
    if not os.path.isdir(write_path):
        os.makedirs(write_path)

    n_batches = len(data_loader)
    shape_data = data_set.data.shape
    n_samp = shape_data[0]
    # labels_unique = data_set.classes    # list of original labels
    label_index_map = data_set.class_to_idx
    exclude_labels = set([label_index_map[a] for a in EXCLUDE_LABELS])

    data_all = []
    targets_all = []
    for i, (data, targets) in enumerate(data_loader):
        data, targets = data.to(device), targets.to(device)
        data_all.append(data.cpu().numpy())
        targets_all.append(targets.cpu().numpy())

    data_all = np.concatenate(data_all, axis=0)
    targets_all = np.concatenate(targets_all)
    # Include only a subset of the labels
    ind_incl = np.array([i for i in range(targets_all.shape[0]) if targets_all[i] not in exclude_labels])
    data_all = data_all[ind_incl, :]
    targets_all = targets_all[ind_incl]
    labels_unique, counts_unique = np.unique(targets_all, return_counts=True)
    n_samp_per_class = dict(zip(labels_unique, counts_unique))
    print("Number of labels included: {:d}".format(labels_unique.shape[0]))

    fname = os.path.join(write_path, 'label_index_mapping{}.csv'.format(suffix))
    with open(fname, 'w') as fp:
        cw = csv.writer(fp, delimiter=',', lineterminator='\n')
        cw.writerow(['label', 'label_index', 'n_samples'])
        for k, v in label_index_map.items():
            if v not in exclude_labels:
                cw.writerow([k, '{:d}'.format(v), n_samp_per_class[v]])

    # Save the data and target arrays to .npy files
    fname = os.path.join(write_path, 'data{}.npy'.format(suffix))
    with open(fname, 'wb') as fp:
        np.save(fp, data_all)

    fname = os.path.join(write_path, 'labels{}.npy'.format(suffix))
    with open(fname, 'wb') as fp:
        np.save(fp, targets_all)

    if cross_val:
        # Do a stratified cross-validation split and save the train and test fold data to numpy files
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED_DEFAULT)
        fold = 1
        for index_tr, index_te in skf.split(data_all, targets_all):
            final_write_path = os.path.join(write_path, 'fold_' + str(fold))
            if not os.path.isdir(final_write_path):
                os.makedirs(final_write_path)

            data_tr, data_te = data_all[index_tr, :], data_all[index_te, :]
            labels_tr, labels_te = targets_all[index_tr], targets_all[index_te]
            np.save(os.path.join(final_write_path, 'data_tr.npy'), data_tr)
            np.save(os.path.join(final_write_path, 'labels_tr.npy'), labels_tr)
            np.save(os.path.join(final_write_path, 'data_te.npy'), data_te)
            np.save(os.path.join(final_write_path, 'labels_te.npy'), labels_te)
            fold = fold + 1


def main():
    torch.manual_seed(SEED_DEFAULT)
    use_cuda = USE_GPU and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Train and test sets for the CIFAR-100 dataset
    data_path = DATA_PATH
    write_path = os.path.join(NUMPY_DATA_PATH, 'cifar100')
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(*NORMALIZE_IMAGES['cifar10'])]
    )
    # Using the same transformations for the training and test images
    train_set = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_test)
    test_set = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
    # Data loaders for the train and test set
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    process_data(test_set, test_loader, device, write_path, True, suffix='_test')
    process_data(train_set, train_loader, device, write_path, False, suffix='_train')


if __name__ == '__main__':
    main()
