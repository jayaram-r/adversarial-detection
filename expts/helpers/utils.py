
import numpy as np
import torch
import os
from multiprocessing import cpu_count
from helpers.generate_data import MFA_model
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    average_precision_score
)
from helpers.constants import ROOT
from torch.utils.data import TensorDataset, DataLoader


def convert_to_list(array):
    #array is a numpy ndarray
    return [r for r in array]


def convert_to_loader(x, y, batch_size=1):
    # transform to torch tensor; using `as_tensor` avoids creating a copy
    #tensor_x = torch.as_tensor(x)
    tensor_x = torch.tensor(x)
    #tensor_y = torch.as_tensor(y)
    tensor_y = torch.tensor(y)

    # create your dataset and data loader
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size)


def get_samples_as_ndarray(loader):
    X = []
    Y = []
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.cpu().numpy(), target.cpu().numpy()
        target = target.reshape((target.shape[0], 1))
        if batch_idx == 0:
            X, Y = data, target
        else:
            X = np.vstack((X,data))
            Y = np.vstack((Y,target))

    Y = Y.reshape((-1,))
    return X,Y


def verify_data_loader(loader, batch_size=1):
    X_1, Y_1 = get_samples_as_ndarray(loader)
    loader_new = convert_to_loader(X_1, Y_1, batch_size=batch_size)
    X_2, Y_2 = get_samples_as_ndarray(loader_new)

    return np.array_equal(X_1, X_2) and np.array_equal(Y_1, Y_2)


def load_numpy_data(path, adversarial=False):
    print("Loading saved numpy data from the path:", path)
    if not adversarial:
        data_tr = np.load(os.path.join(path, "data_tr.npy"))
        labels_tr = np.load(os.path.join(path, "labels_tr.npy"))
        data_te = np.load(os.path.join(path, "data_te.npy"))
        labels_te = np.load(os.path.join(path, "labels_te.npy"))
    else:
        data_tr = np.load(os.path.join(path, "data_tr_adv.npy"))
        labels_tr = np.load(os.path.join(path, "labels_tr_adv.npy"))
        data_te = np.load(os.path.join(path, "data_te_adv.npy"))
        labels_te = np.load(os.path.join(path, "labels_te_adv.npy"))

    return data_tr, labels_tr, data_te, labels_te


def get_data_bounds(data):
    bounds = [np.min(data), np.max(data)]
    # Value slightly smaller than the minimum
    bounds[0] = 0.999 * bounds[0] if (bounds[0] >= 0) else 1.001 * bounds[0]
    # Value slightly larger than the maximum
    bounds[1] = 1.001 * bounds[1] if (bounds[1] >= 0) else 0.999 * bounds[1]

    return tuple(bounds)


def get_model_file(model_type, epoch=None):
    if epoch is None:
        return os.path.join(ROOT, 'models', '{}_cnn.pt'.format(model_type))
    else:
        return os.path.join(ROOT, 'models', '{}_epoch_{}_cnn.pt'.format(model_type, epoch))


def load_model_checkpoint(model, model_type, epoch=None):
    model_path = get_model_file(model_type, epoch=epoch)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        raise InputError("Saved model checkpoint '{}' not found.".format(model_path))

    return model


def save_model_checkpoint(model, model_type, epoch=None):
    model_path = get_model_file(model_type, epoch=epoch)
    torch.save(model.state_dict(), model_path)


def get_path_dr_models(model_type):
    # Default path to the dimensionality reduction model files
    return os.path.join(
        ROOT, 'models', 'models_dimension_reduction', model_type, 'models_dimension_reduction.pkl'
    )


def get_clean_data_path(model_type, fold):
    return os.path.join(ROOT, 'numpy_data', model_type, 'fold_{}'.format(fold))


def get_adversarial_data_path(model_type, fold, attack_type, attack_param_list):
    # Example `attack_param_list = [('stepsize', 0.05), ('confidence', 0), ('epsilon', 0.0039)]`
    base = ''.join(['{}_{}'.format(a, str(b)) for a, b in attack_param_list])
    return os.path.join(ROOT, 'numpy_data', model_type, 'fold_{}'.format(fold), attack_type, base)


def get_output_path(model_type):
    # Default output path
    return os.path.join(ROOT, 'outputs', model_type)


def metrics_detection(scores, labels, pos_label=1, max_fpr=0.01, verbose=True):
    """
    Wrapper function that calculates a bunch of performance metrics for anomaly detection.

    :param scores: numpy array with the anomaly scores. Larger values correspond to higher probability of a
                   point being anomalous.
    :param labels: numpy array of labels indicating whether a point is nominal (value 0) or anomalous (value 1).
    :param pos_label: value corresponding to the anomalous class in `labels`.
    :param max_fpr: float value in `(0, 1)`. The partial area under the ROC curve is calculated for
                    FPR <= this value.
    :param verbose: Set to True to print the performance metrics.
    :return:
    """
    # 0.1%, 0.5%, 1%, 5%, and 10%
    fpr_thresh = [0.001, 0.005, 0.01, 0.05, 0.1]
    au_roc = roc_auc_score(labels, scores)
    au_roc_partial = roc_auc_score(labels, scores, max_fpr=max_fpr)
    avg_prec = average_precision_score(labels, scores)

    if verbose:
        print("Area under the ROC curve = {:.6f}".format(au_roc))
        print("Partial area under the ROC curve (FPR <= {:.2f}) = {:.6f}".format(max_fpr, au_roc_partial))
        print("Average precision = {:.6f}".format(avg_prec))

    # ROC curve and TPR at a few low FPR values
    fpr_arr, tpr_arr, thresh = roc_curve(labels, scores, pos_label=pos_label)
    tpr = np.zeros(len(fpr_thresh))
    fpr = np.zeros_like(tpr)
    if verbose:
        print("\nTPR, FPR")

    for i, a in enumerate(fpr_thresh):
        mask = fpr_arr >= a
        tpr[i] = tpr_arr[mask][0]
        fpr[i] = fpr_arr[mask][0]
        if verbose:
            print("{:.6f}, {:.6f}".format(tpr[i], fpr[i]))

    return au_roc, au_roc_partial, avg_prec, tpr, fpr


def get_num_jobs(n_jobs):
    """
    Number of processes or jobs to use for multiprocessing.

    :param n_jobs: None or int value that specifies the number of parallel jobs. If set to None, -1, or 0, this will
                   use all the available CPU cores. If set to negative values, this value will be subtracted from
                   the available number of CPU cores. For example, `n_jobs = -2` will use `cpu_count - 2`.
    :return: (int) number of jobs to use.
    """
    cc = cpu_count()
    if n_jobs is None or n_jobs == -1 or n_jobs == 0:
        n_jobs = cc
    elif n_jobs < -1:
        n_jobs = max(1, cc + n_jobs)
    else:
        n_jobs = min(n_jobs, cc)

    return n_jobs


def wrapper_data_generate(dim, dim_latent_range, n_components, N_train, N_test,
                          prop_anomaly=0.1, anom_type='uniform', seed_rng=123):
    """
    A wrapper function to generate synthetic training and test data for anomaly detection. Nominal data is generated
    from a mixture of factor analyzers (MFA) model. Anomalous data are generated according to a uniform or Gaussian
    distribution, independently for each feature.

    :param dim: (int) dimension or number of features.
    :param dim_latent_range: (tuple) range of the latent dimension specified as a tuple of two int values.
    :param n_components: (int) number of mixture components in the MFA model.
    :param N_train: (int) number of train samples.
    :param N_test: (int) number of test samples.
    :param prop_anomaly: (float) proportion of test samples that are anomalous. Value should be in (0, 1).
    :param anom_type: 'uniform' or 'gaussian'.
    :param seed_rng: (int) seed for the random number generator.

    :return: (data, data_test, labels_test)
        - data: numpy array of shape `(N_train, dim)` with the nominal data points for training.
        - data_test: numpy array of shape `(N_test, dim)` with the test data points. Containts a mix of nominal
                     and anomalous data.
    """
    # Generate data according to a mixture of factor analysis (MFA) model
    model = MFA_model(n_components, dim, dim_latent_range=dim_latent_range, seed_rng=seed_rng)

    # Generate nominal data from the MFA model
    data, _ = model.generate_data(N_train)

    # Generate a mixture of nominal and anomalous data as the test set
    N_test_anom = int(np.ceil(prop_anomaly * N_test))
    N_test_nom = N_test - N_test_anom
    data_nom, _ = model.generate_data(N_test_nom)

    # Anomalous points are generated either from a uniform distribution or a Gaussian distribution
    x_min = np.min(data, axis=0)
    x_max = np.max(data, axis=0)
    x_mean = np.mean(data, axis=0)
    x_std = np.std(data, axis=0)

    data_anom = np.zeros((N_test_anom, dim))
    for i in range(dim):
        if anom_type == 'uniform':
            data_anom[:, i] = np.random.uniform(low=x_min[i], high=x_max[i], size=N_test_anom)
        else:
            # Gaussian
            data_anom[:, i] = np.random.normal(loc=x_mean[i], scale=x_std[i], size=N_test_anom)

    data_test = np.concatenate([data_nom, data_anom], axis=0)
    labels_test = np.concatenate([np.zeros(N_test_nom, dtype=np.int),
                                  np.ones(N_test_anom, dtype=np.int)])

    return data, data_test, labels_test
