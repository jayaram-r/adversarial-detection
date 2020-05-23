
import numpy as np
import torch
import os
import sys
import pickle
from multiprocessing import cpu_count
from helpers.generate_data import MFA_model
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    average_precision_score
)
from helpers.constants import (
    ROOT,
    NUMPY_DATA_PATH,
    SEED_DEFAULT,
    FPR_MAX_PAUC,
    FPR_THRESH,
    BATCH_SIZE_DEF,
    COLORS,
    MARKERS
)
from torch.utils.data import Dataset, TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]

        return data, target, index

    def __len__(self):
        return len(self.dataset)


def convert_to_loader(x, y, dtype_x=None, dtype_y=None, device=None, batch_size=BATCH_SIZE_DEF, shuffle=False,
                      custom=False):
    # Setting `custom = True` gives a data loader that also returns the index of samples in the batch
    # transform to torch tensors
    x_ten = torch.tensor(x, dtype=dtype_x, device=device)
    y_ten = torch.tensor(y, dtype=dtype_y, device=device)
    # create the dataset and data loader
    dataset = TensorDataset(x_ten, y_ten)
    if custom:
        dataset = MyDataset(dataset)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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


def get_samples_as_ndarray(loader):
    X = np.array([])
    Y = np.array([])
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.cpu().numpy(), target.cpu().numpy()
        target = target.reshape((target.shape[0], 1))
        if batch_idx == 0:
            X, Y = data, target
        else:
            X = np.vstack((X, data))
            Y = np.vstack((Y, target))

    return X, Y.ravel()


def verify_data_loader(loader, batch_size=1):
    X_1, Y_1 = get_samples_as_ndarray(loader)
    loader_new = convert_to_loader(X_1, Y_1, batch_size=batch_size)
    X_2, Y_2 = get_samples_as_ndarray(loader_new)

    return np.array_equal(X_1, X_2) and np.array_equal(Y_1, Y_2)


def check_label_mismatch(labels, labels_pred, frac=1.0):
    d = np.sum(labels != labels_pred) / float(labels.shape[0])
    if d < frac:
        print("Fraction of mismatched samples with different original and predicted labels = {:.4f}".format(d))


def load_numpy_data(path):
    # Utility to load clean data and labels from saved numpy files
    data_tr = np.load(os.path.join(path, "data_tr.npy"))
    labels_tr = np.load(os.path.join(path, "labels_tr.npy"))
    data_te = np.load(os.path.join(path, "data_te.npy"))
    labels_te = np.load(os.path.join(path, "labels_te.npy"))

    return data_tr, labels_tr, data_te, labels_te


def load_adversarial_data(path, max_n_test=None, sampling_type='ranked_by_norm', norm_type='inf', seed=SEED_DEFAULT):
    """
    Utility to load adversarial data and labels from saved numpy files. Allows the number of samples from the test
    fold to be specified and has a few sampling options.

    :param path: path to the directory where the numpy files are saved.
    :param max_n_test: None or an int value specifying the maximum number of samples in the test data fold.
    :param sampling_type: string specifying the type of sampling to use. Valid values are:
                         ('uniform', 'ranked_by_norm', 'importance')
    :param norm_type: string or int value specifying the type of norm. Valid values are 'inf' and non-negative
                      integer values.
    :param seed: seed for the random number generator.
    :return:
    """
    # Adversarial inputs from the train and test fold
    data_tr_adv = np.load(os.path.join(path, "data_tr_adv.npy"))
    data_te_adv = np.load(os.path.join(path, "data_te_adv.npy"))

    # Clean inputs corresponding to the adversarial inputs from the train and test fold
    data_tr_clean = np.load(os.path.join(path, "data_tr_clean.npy"))
    data_te_clean = np.load(os.path.join(path, "data_te_clean.npy"))

    # Predicted (mis-classified) labels
    labels_pred_tr = np.load(os.path.join(path, "labels_tr_adv.npy"))
    labels_pred_te = np.load(os.path.join(path, "labels_te_adv.npy"))
    
    # Labels of the original clean inputs from which the adversarial inputs were created
    labels_tr = np.load(os.path.join(path, "labels_tr_clean.npy"))
    labels_te = np.load(os.path.join(path, "labels_te_clean.npy"))

    # Check if the original and adversarial labels are all different
    check_label_mismatch(labels_tr, labels_pred_tr)
    check_label_mismatch(labels_te, labels_pred_te)

    n_test = labels_te.shape[0]
    all_test = True
    if max_n_test:
        if n_test > max_n_test:
            all_test = False

    if all_test:
        # First two returned values are the clean data arrays from the train and test fold.
        # The rest are adversarial data and label arrays from the train and test fold
        return data_tr_clean, data_te_clean, data_tr_adv, labels_tr, data_te_adv, labels_te
    else:
        # Test fold data is sub-sampled to the required size using one of the sampling methods
        np.random.seed(seed)
        ind_test = None
        if sampling_type == 'uniform':
            print("Number of adversarial samples from the test fold = {:d}. Selecting a random subset of {:d} "
                  "samples\n".format(n_test, max_n_test))
            ind_test = np.random.permutation(n_test)[:max_n_test]
        else:
            # Calculate the norm of the perturbation for adversarial inputs from the test fold
            diff = data_te_adv.reshape(n_test, -1) - data_te_clean.reshape(n_test, -1)
            if norm_type == 'inf':
                norm_diff = np.linalg.norm(diff, ord=np.inf, axis=1)
            else:
                # expecting a non-negative integer
                norm_diff = np.linalg.norm(diff, ord=int(norm_type), axis=1)

            if sampling_type == 'ranked_by_norm':
                print("Number of adversarial samples from the test fold = {:d}. Selecting the subset of {:d} samples"
                      " with the smallest perturbation {}-norm\n".format(n_test, max_n_test, norm_type))
                # Select the `max_n_test` samples with the least norm
                ind_test = np.argsort(norm_diff)[:max_n_test]

            elif sampling_type == 'importance':
                print("Number of adversarial samples from the test fold = {:d}. Selecting a random subset of {:d} "
                      "samples with importance sampling based on the inverse perturbation {}-norm\n".
                      format(n_test, max_n_test, norm_type))
                # Assign each test sample a weight proportional to the inverse of its perturbation norm and
                # do importance sampling to select `max_n_test` samples (without replacement)
                samp_wt = 1. / np.clip(norm_diff, sys.float_info.epsilon, None)
                p = samp_wt / np.sum(samp_wt)
                ind_test = np.random.choice(n_test, size=max_n_test, replace=False, p=p)

            else:
                raise ValueError("Invalid value '{}' specified for the input 'sampling_type'".format(sampling_type))

        return (data_tr_clean, data_te_clean[ind_test, :], data_tr_adv, labels_tr, data_te_adv[ind_test, :],
                labels_te[ind_test])


def load_noisy_data(path):
    # Utility to load noisy data and labels from saved numpy files
    data_tr = None
    data_te = None
    k = 0
    for file_pre in ('data_tr_noisy_stdev_', 'data_te_noisy_stdev_'):
        len_pre = len(file_pre)
        stdev_list = []
        data_list = []
        for f in os.listdir(path):
            if f.startswith(file_pre):
                # Get the standard deviation value from the filename
                a = os.path.splitext(f)[0]
                stdev_list.append(float(a[len_pre:]))
                data_list.append(np.load(os.path.join(path, f)))

        print("Noise standard deviation values:")
        print(', '.join(['{:.6f}'.format(v) for v in stdev_list]))
        n_vals = len(stdev_list)
        n_samp = data_list[0].shape[0]
        if not all([data_list[i].shape[0] == n_samp for i in range(n_vals)]):
            raise ValueError("Data arrays for different noise standard deviations have different sizes")

        # Each noisy sample is selected randomly from one of the noise standard deviation values
        ind_cols = np.random.choice(np.arange(n_vals), size=n_samp, replace=True)
        data = np.array([data_list[ind_cols[i]][i, :] for i in range(n_samp)])
        if k == 0:
            data_tr = data
        else:
            data_te = data

        k += 1

    return data_tr, data_te


def get_data_bounds(data, alpha=0.95):
    bounds = [data.min(), data.max()]
    # Value slightly smaller than the minimum
    bounds[0] = (alpha * bounds[0]) if bounds[0] >= 0 else (bounds[0] / alpha)
    # Value slightly larger than the maximum
    bounds[1] = (bounds[1] / alpha) if bounds[1] >= 0 else (alpha * bounds[1])

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


def get_path_dr_models(model_type, method_detection, test_statistic=None):
    # Path to the dimensionality reduction model files. Different models are used depending on the method
    fname1 = os.path.join(
        ROOT, 'models', 'models_dimension_reduction', model_type, 'models_dimension_reduction.pkl'
    )
    fname2 = os.path.join(
        ROOT, 'models', 'models_dimension_reduction', model_type, 'models_fixed_dimension_1000.pkl'
    )
    fname3 = os.path.join(
        ROOT, 'models', 'models_dimension_reduction', model_type, 'models_fixed_dimension_1000_lid.pkl'
    )
    fname = fname1
    if method_detection == 'proposed':
        if test_statistic in ['lid', 'lle']:
            # Not a mistake. Projecting to a fixed dimension for the LID and LLE methods (using `fname2`) does
            # not work well
            fname = fname1
        else:
            fname = fname1

    elif method_detection in ['lid', 'lid_class_cond']:
        # This method uses more layers
        fname = fname3
    else:
        fname = fname1

    return fname


def get_clean_data_path(model_type, fold):
    return os.path.join(NUMPY_DATA_PATH, model_type, 'fold_{}'.format(fold))


def get_noisy_data_path(model_type, fold):
    return os.path.join(NUMPY_DATA_PATH, model_type, 'fold_{}'.format(fold), 'noise_gaussian')


def get_adversarial_data_path(model_type, fold, attack_type, attack_param_list):
    # Example `attack_param_list = [('stepsize', 0.05), ('confidence', 0), ('epsilon', 0.0039)]`
    base = ''.join(['{}_{}'.format(a, str(b)) for a, b in attack_param_list])
    return os.path.join(NUMPY_DATA_PATH, model_type, 'fold_{}'.format(fold), attack_type, base)


def get_output_path(model_type):
    # Default output path
    return os.path.join(ROOT, 'outputs', model_type)


def list_all_adversarial_subdirs(model_type, fold, attack_type, check_subdirectories=True):
    # List all sub-directories corresponding to an adversarial attack
    d = os.path.join(NUMPY_DATA_PATH, model_type, 'fold_{}'.format(fold), attack_type)
    # Temporary hack to use backup data directory
    d = d.replace('varun', 'jayaram', 1)
    if not os.path.isdir(d):
        raise ValueError("Directory '{}' does not exist.".format(d))
    
    if check_subdirectories:
        d_sub = [os.path.join(d, f) for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))]
        if not d_sub:
            raise ValueError("Directory '{}' does not have any sub-directories.".format(d))

        return sorted(d_sub)
    else:
        # This handles the custom attack which does not have any parameter-specific subdirectories
        return [d]


def save_detector_checkpoint(scores_folds, labels_folds, models_folds, output_dir, method_name, save_detec_model):
    # Save the scores and detection labels from the cross-validation folds to a pickle file
    fname = os.path.join(output_dir, 'scores_{}.pkl'.format(method_name))
    tmp = {'scores_folds': scores_folds, 'labels_folds': labels_folds}
    with open(fname, 'wb') as fp:
        pickle.dump(tmp, fp)

    if save_detec_model and models_folds:
        # Save the detection models from the cross-validation folds to a pickle file
        fname = os.path.join(output_dir, 'models_{}.pkl'.format(method_name))
        with open(fname, 'wb') as fp:
            pickle.dump(models_folds, fp)


def load_detector_checkpoint(output_dir, method_name, save_detec_model):
    fname = os.path.join(output_dir, 'scores_{}.pkl'.format(method_name))
    with open(fname, 'rb') as fp:
        tmp = pickle.load(fp)

    scores_folds = tmp['scores_folds']
    labels_folds = tmp['labels_folds']
    # Load the models if required
    if save_detec_model:
        fname = os.path.join(output_dir, 'models_{}.pkl'.format(method_name))
        with open(fname, 'rb') as fp:
            models_folds = pickle.load(fp)
    else:
        models_folds = []

    n_folds = len(scores_folds)
    assert len(labels_folds) == n_folds, "'scores_folds' and 'labels_folds' do not have the same length"
    if models_folds:
        assert len(models_folds) == n_folds, "'models_folds' and 'scores_folds' do not have the same length"

    return scores_folds, labels_folds, models_folds, n_folds


def calculate_accuracy(model, device, data_loader=None, data=None, labels=None, batch_size=BATCH_SIZE_DEF):
    """
    Calculate the accuracy of a trained model on a given data set.

    :param model: trained model object.
    :param device: torch device object.
    :param data_loader: None or a torch data loader. If this is specified, the inputs `data` and `labels` will
                        be ignored.
    :param data: None or a numpy array of inputs.
    :param labels: None or a numpy array of labels (targets).
    :param batch_size: batch size for prediction.

    :return: model accuracy in percentage.
    """
    if model.training:
        model.eval()

    if data_loader is None:
        if (data is None) or (labels is None):
            raise ValueError("Invalid inputs - data and/or labels are not specified.")

        # Create a torch data loader from numpy arrays
        data_loader = convert_to_loader(data, labels, dtype_x=torch.float, device=device, batch_size=batch_size)
        n_samp = labels.shape[0]
    else:
        n_samp = len(data_loader.dataset)

    correct = 0.
    with torch.no_grad():
        for data_bt, target_bt in data_loader:
            output = model(data_bt)
            _, predicted = output.max(1)
            correct += predicted.eq(target_bt).sum().item()

    return (100. * correct) / n_samp


def get_predicted_classes(model, device, data_loader=None, data=None, labels=None, batch_size=BATCH_SIZE_DEF):
    """
    Get the model-predicted classes on a given data set.

    :param model: trained model object.
    :param device: torch device object.
    :param data_loader: None or a torch data loader. If this is specified, the inputs `data` and `labels` will
                        be ignored.
    :param data: None or a numpy array of inputs.
    :param labels: None or a numpy array of labels (targets).
    :param batch_size: batch size for prediction.

    :return: numpy array of predicted classes.
    """
    if model.training:
        model.eval()

    if data_loader is None:
        if (data is None) or (labels is None):
            raise ValueError("Invalid inputs - data and/or labels are not specified.")

        # Create a torch data loader from numpy arrays
        data_loader = convert_to_loader(data, labels, dtype_x=torch.float, device=device, batch_size=batch_size)

    labels_pred = []
    with torch.no_grad():
        for data_bt, target_bt in data_loader:
            output = model(data_bt)
            _, predicted = output.max(1)
            labels_pred.extend(predicted.detach().cpu().numpy())

    return np.array(labels_pred, dtype=np.int)


def add_gaussian_noise(data_in, stdev_values, seed=SEED_DEFAULT):
    # Add random Gaussian noise from a given list of standard deviation values to the input data
    np.random.seed(seed)
    shape_data = data_in.shape
    n_samp = shape_data[0]
    n_vals = len(stdev_values)

    # Generate a noisy version of the input data for each standard deviation value
    data_noisy_list = []
    for sig in stdev_values:
        noise = np.random.normal(loc=0., scale=sig, size=shape_data)
        data_noisy_list.append(data_in + noise)

    # Each noisy sample is selected randomly from one of the noise standard deviation values
    ind_cols = np.random.choice(np.arange(n_vals), size=n_samp, replace=True)
    data_out = np.array([data_noisy_list[ind_cols[i]][i, :] for i in range(n_samp)])

    return data_out


def log_sum_exp(x):
    # Numerically stable computation of the log-sum-exponential function
    # `x` is a 2d numpy array
    # Sum is across the columns of `x`
    if x.shape[1] > 1:
        col_max = np.max(x, axis=1)
        v = np.sum(np.exp(x - col_max[:, np.newaxis]), axis=1)
        return np.log(np.clip(v, sys.float_info.min, None)) + col_max
    elif x.shape[1] == 1:
        return x[:, 0]
    else:
        return x


def metrics_detection(scores, labels, pos_label=1, max_fpr=FPR_MAX_PAUC, verbose=True):
    """
    Wrapper function that calculates a bunch of performance metrics for anomaly detection.

    :param scores: numpy array with the anomaly scores. Larger values correspond to higher probability of a
                   point being anomalous.
    :param labels: numpy array of labels indicating whether a point is nominal (value 0) or anomalous (value 1).
    :param pos_label: value corresponding to the anomalous class in `labels`.
    :param max_fpr: float or an iterable of float values in `(0, 1)`. The partial area under the ROC curve is
                    calculated for each FPR value in `max_fpr`.
    :param verbose: Set to True to print the performance metrics.
    :return:
    """
    au_roc = roc_auc_score(labels, scores)
    avg_prec = average_precision_score(labels, scores)
    if hasattr(max_fpr, '__iter__'):
        au_roc_partial = np.array([roc_auc_score(labels, scores, max_fpr=v) for v in max_fpr])
    else:
        au_roc_partial = roc_auc_score(labels, scores, max_fpr=max_fpr)

    if verbose:
        print("Area under the ROC curve = {:.6f}".format(au_roc))
        print("Average precision = {:.6f}".format(avg_prec))
        print("Partial area under the ROC curve (pauc):")
        if hasattr(au_roc_partial, '__iter__'):
            for a, b in zip(max_fpr, au_roc_partial):
                print("pauc below fpr {:.4f} = {:.6f}".format(a, b))
        else:
            print("pauc below fpr {:.4f} = {:.6f}".format(max_fpr, au_roc_partial))

    # ROC curve and TPR at a few low FPR values
    fpr_arr, tpr_arr, thresh = roc_curve(labels, scores, pos_label=pos_label)
    tpr = np.zeros(len(FPR_THRESH))
    fpr = np.zeros_like(tpr)
    if verbose:
        print("\nTPR, FPR")

    for i, a in enumerate(FPR_THRESH):
        mask = fpr_arr >= a
        tpr[i] = tpr_arr[mask][0]
        fpr[i] = fpr_arr[mask][0]
        if verbose:
            print("{:.6f}, {:.6f}".format(tpr[i], fpr[i]))

    return au_roc, au_roc_partial, avg_prec, tpr, fpr


def metrics_for_accuracy(labels, clean_labels, output_file=None):
    
    # Save the results to a pickle file if required
    if output_file:
        with open(output_file, 'wb') as fp:
            pickle.dump(results, fp)

    return results


def metrics_varying_positive_class_proportion(scores, labels, pos_label=1, num_prop=10,
                                              num_random_samples=100, seed=SEED_DEFAULT, output_file=None,
                                              max_pos_proportion=1.0, log_scale=False):
    """
    Calculate a number of performance metrics as the fraction of positive samples in the data is varied.
    For each proportion, the estimates are calculated from different random samples, and the median and confidence
    interval values of each performance metric are reported.

    :param scores: list of 1D numpy arrays with the detection scores from the test folds of cross-validation.
    :param labels: list of 1D numpy array with the binary detection labels from the test folds of cross-validation.
    :param pos_label: postive class label; set to 1 by default.
    :param num_prop: number of positive proportion values to evaluate.
    :param num_random_samples: number of random samples to use for estimating the median and confidence interval.
    :param seed: seed for the random number generator.
    :param output_file: (optional) path to an output file where the metrics dict is written to using the
                        Pickle protocol.
    :param max_pos_proportion: Maximum proportion of positive samples to include in the plots. Should be a float
                               value between 0 and 1.
    :param log_scale: Set to True to use logarithmically spaced positive proportion values.

    :return: a dict with the proportion of positive samples and all the performance metrics.
    """
    np.random.seed(seed)
    n_folds = len(scores)
    n_samp = []
    ind_pos = []
    n_pos_max = []
    scores_neg = []
    labels_neg = []
    for i in range(n_folds):
        n_samp.append(float(labels[i].shape[0]))
        # index of positive labels
        mask = (labels[i] == pos_label)
        temp = np.where(mask)[0]
        ind_pos.append(temp)
        n_pos_max.append(temp.shape[0])
        # index of negative labels
        temp = np.where(~mask)[0]
        scores_neg.append(scores[i][temp])
        labels_neg.append(labels[i][temp])

    # Minimum proportion of positive samples. Ensuring that there are at least 5 positive samples
    p_min = max([max(5., np.ceil(0.005 * n_samp[i])) / n_samp[i] for i in range(n_folds)])
    # Maximum proportion of positive samples considering all the folds
    p_max = min([n_pos_max[i] / n_samp[i] for i in range(n_folds)])
    p_max = min(p_max, max_pos_proportion)
    # Range of proportion of positive samples
    if log_scale:
        prop_range = np.unique(np.logspace(np.log10(p_min), np.log10(p_max), num=num_prop))
    else:
        prop_range = np.unique(np.linspace(p_min, p_max, num=num_prop))

    # dict with the positive proportion and the corresponding performance metrics. The mean, and lower and upper
    # confidence interval values are calculated for each performance metric
    results = {
        'proportion': prop_range,
        'auc': {'median': [], 'CI_lower': [], 'CI_upper': []},
        'avg_prec': {'median': [], 'CI_lower': [], 'CI_upper': []},
        'pauc': {'median': [], 'CI_lower': [], 'CI_upper': []},
        'tpr': {'median': [], 'CI_lower': [], 'CI_upper': []},
        'fpr': {'median': [], 'CI_lower': [], 'CI_upper': []}
    }
    metric_names = list(results.keys())
    metric_names.remove('proportion')

    ##################### A small utility function
    def _append_percentiles(x, y):
        if len(x.shape) == 2:
            p = np.percentile(x, [2.5, 50, 97.5], axis=0)
            a = p[0, :]
            b = p[1, :]
            c = p[2, :]
        else:
            a, b, c = np.percentile(x, [2.5, 50, 97.5])

        y['CI_lower'].append(a)
        y['median'].append(b)
        y['CI_upper'].append(c)

        return a, b, c
    #####################

    num_pauc = len(FPR_MAX_PAUC)
    num_tpr = len(FPR_THRESH)
    # Varying the proportion of positive samples
    for p in prop_range:
        print("\nPerformance metrics for target positive proportion: {:.4f}".format(p))

        metrics_dict = {k: [] for k in metric_names}
        # Cross-validation folds
        for i in range(n_folds):
            # number of positive samples from this fold
            n_pos = min(int(np.ceil(p * n_samp[i])), n_pos_max[i])
            sample_indices = None
            if n_pos == 1:
                if n_pos_max[i] > num_random_samples:
                    temp = np.random.permutation(ind_pos[i])[:num_random_samples]
                    sample_indices = temp[:, np.newaxis]
                    t = num_random_samples
                else:
                    sample_indices = ind_pos[i][:, np.newaxis]
                    t = n_pos_max[i]

            elif n_pos >= (n_pos_max[i] - 1):
                # Include all the positive indices in 1 sample
                n_pos = n_pos_max[i]
                t = 1
                sample_indices = ind_pos[i][np.newaxis, :]
            else:
                t = num_random_samples

            print("Fold {:d}: Number of positive samples = {:d}. Target proportion = {:.4f}. Actual proportion"
                  " = {:.4f}".format(i + 1, n_pos, p, n_pos / n_samp[i]))
            # Repeating over `t` randomly selected positive subsets
            auc_curr = np.zeros(t)
            pauc_curr = np.zeros((t, num_pauc))
            ap_curr = np.zeros(t)
            tpr_curr = np.zeros((t, num_tpr))
            fpr_curr = np.zeros((t, num_tpr))
            for j in range(t):
                if sample_indices is None:
                    ind_curr = np.random.permutation(ind_pos[i])[:n_pos]
                else:
                    ind_curr = sample_indices[j, :]

                scores_curr = np.concatenate([scores[i][ind_curr], scores_neg[i]])
                labels_curr = np.concatenate([labels[i][ind_curr], labels_neg[i]])
                # Calculate performance metrics for this trial
                ret = metrics_detection(scores_curr, labels_curr, pos_label=pos_label, verbose=False)
                auc_curr[j] = ret[0]
                pauc_curr[j, :] = ret[1]
                ap_curr[j] = ret[2]
                tpr_curr[j, :] = ret[3]
                fpr_curr[j, :] = ret[4]

            metrics_dict['auc'].append(auc_curr)
            metrics_dict['pauc'].append(pauc_curr)
            metrics_dict['avg_prec'].append(ap_curr)
            metrics_dict['tpr'].append(tpr_curr)
            metrics_dict['fpr'].append(fpr_curr)

        # Concatenate the performance metrics from the different folds.
        # Then calculate the median, 2.5 and 97.5 percentile of each performance metric
        arr = np.concatenate(metrics_dict['auc'], axis=0)
        ret = _append_percentiles(arr, results['auc'])
        print("Area under the ROC curve = {:.6f}".format(ret[1]))

        arr = np.concatenate(metrics_dict['avg_prec'], axis=0)
        ret = _append_percentiles(arr, results['avg_prec'])
        print("Average precision = {:.6f}".format(ret[1]))

        arr = np.concatenate(metrics_dict['pauc'], axis=0)
        ret = _append_percentiles(arr, results['pauc'])
        for a, b in zip(FPR_MAX_PAUC, ret[1]):
            print("Partial-AUC below fpr {:.4f} = {:.6f}".format(a, b))

        arr = np.concatenate(metrics_dict['tpr'], axis=0)
        ret1 = _append_percentiles(arr, results['tpr'])
        arr = np.concatenate(metrics_dict['fpr'], axis=0)
        ret2 = _append_percentiles(arr, results['fpr'])
        print("TPR\tFPR_target\tFPR_actual\tTPR_scaled")
        for a, b, c in zip(ret1[1], FPR_THRESH, ret2[1]):
            print("{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(a, b, c, a / max(1, c / b)))

    # Save the results to a pickle file if required
    if output_file:
        with open(output_file, 'wb') as fp:
            pickle.dump(results, fp)

    return results


def plot_helper(plot_dict, methods, plot_file, min_yrange=None, place_legend_outside=False,
                log_scale=False, n_ticks=8, hide_errorbar=False):
    fig = plt.figure()
    if log_scale:
        plt.xscale('log', basex=10)

    x_vals = []
    y_vals = []
    for j, m in enumerate(methods):
        d = plot_dict[m]
        if hide_errorbar or ('y_err' not in d):
            plt.plot(d['x_vals'], d['y_vals'], linestyle='--', color=COLORS[j], marker=MARKERS[j], label=m)
        else:
            plt.errorbar(d['x_vals'], d['y_vals'], yerr=d['y_err'],
                         fmt='', elinewidth=1, capsize=4,
                         linestyle='--', color=COLORS[j], marker=MARKERS[j], label=m)

        x_vals.extend(d['x_vals'])
        y_vals.extend(d['y_vals'])
        if not hide_errorbar:
            if 'y_low' in d:
                y_vals.extend(d['y_low'])
            if 'y_up' in d:
                y_vals.extend(d['y_up'])

    x_bounds = get_data_bounds(np.array(x_vals), alpha=0.99)
    y_bounds = get_data_bounds(np.array(y_vals), alpha=0.99)
    if min_yrange:
        # Ensure that the range of y-axis is not smaller than `min_yrange`
        v = min(y_bounds[1] - min_yrange, y_bounds[0])
        y_bounds = (v, y_bounds[1])

    plt.xlim([x_bounds[0], x_bounds[1]])
    plt.ylim([y_bounds[0], y_bounds[1]])
    plt.yticks(np.linspace(y_bounds[0], y_bounds[1], num=n_ticks), rotation=0)
    if log_scale:
        plt.xticks(np.logspace(np.log10(x_bounds[0]), np.log10(x_bounds[1]), num=n_ticks), rotation=0)
    else:
        plt.xticks(np.linspace(x_bounds[0], x_bounds[1], num=n_ticks), rotation=0)

    plt.xlabel(plot_dict['x_label'], fontsize=9, fontweight='bold')
    plt.ylabel(plot_dict['y_label'], fontsize=9, fontweight='bold')
    # plt.title(plot_dict['title'], fontsize=9, fontweight='bold')
    if not place_legend_outside:
        plt.legend(loc='best', prop={'size': 'xx-small', 'weight': 'bold'})
    else:
        # place the upper right end of the box outside and slightly below the plot axes
        plt.legend(loc='upper right', bbox_to_anchor=(1, -0.07), prop={'size': 'xx-small', 'weight': 'bold'})

    fig.savefig('{}.png'.format(plot_file), dpi=600, bbox_inches='tight', transparent=False)
    fig.savefig('{}.pdf'.format(plot_file), dpi=600, bbox_inches='tight', transparent=False)
    plt.close(fig)


def plot_performance_comparison(results_dict, output_dir, place_legend_outside=True, pos_label='adversarial',
                                log_scale=False, hide_errorbar=False, name_prefix=''):
    """
    Plot the performance comparison for different detection methods.

    :param results_dict: dict mapping each method name to its metrics dict (obtained from the function
                         `metrics_varying_positive_class_proportion`).
    :param output_dir: path to the output directory where the plots are to be saved.
    :param place_legend_outside: Set to True to place the legend outside the plot area.
    :param pos_label: string with the positive class label.
    :param log_scale: Set to True to use a logarithmic scale on the x-axis.
    :param hide_errorbar: Set to True to hide error-bars from the plots.
    :param name_prefix: String with a file prefix.
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    x_label = 'Proportion of {} samples (%)'.format(pos_label)
    methods = sorted(results_dict.keys())
    # AUC plots
    plot_dict = dict()
    plot_dict['x_label'] = x_label
    plot_dict['y_label'] = 'Area under ROC'
    plot_dict['title'] = 'Area under ROC curve'
    for m in methods:
        d = results_dict[m]
        y_med = np.array(d['auc']['median'])
        y_low = np.array(d['auc']['CI_lower'])
        y_up = np.array(d['auc']['CI_upper'])
        plot_dict[m] = {
            'x_vals': 100 * d['proportion'],
            'y_vals': y_med,
            'y_low': y_low,
            'y_up': y_up,
            'y_err': [y_med - y_low, y_up - y_med]
        }

    if name_prefix:
        plot_file = os.path.join(output_dir, '{}_{}'.format(name_prefix, 'auc'))
    else:
        plot_file = os.path.join(output_dir, '{}'.format('auc'))

    plot_helper(plot_dict, methods, plot_file, min_yrange=0.1, place_legend_outside=place_legend_outside,
                log_scale=log_scale, hide_errorbar=hide_errorbar)

    # Average precision plots
    plot_dict = dict()
    plot_dict['x_label'] = x_label
    plot_dict['y_label'] = 'Average precision'
    plot_dict['title'] = 'Average precision (area under PR curve)'
    for m in methods:
        d = results_dict[m]
        y_med = np.array(d['avg_prec']['median'])
        y_low = np.array(d['avg_prec']['CI_lower'])
        y_up = np.array(d['avg_prec']['CI_upper'])
        plot_dict[m] = {
            'x_vals': 100 * d['proportion'],
            'y_vals': y_med,
            'y_low': y_low,
            'y_up': y_up,
            'y_err': [y_med - y_low, y_up - y_med]
        }

    if name_prefix:
        plot_file = os.path.join(output_dir, '{}_{}'.format(name_prefix, 'avg_prec'))
    else:
        plot_file = os.path.join(output_dir, '{}'.format('avg_prec'))

    plot_helper(plot_dict, methods, plot_file, min_yrange=0.1, place_legend_outside=place_legend_outside,
                log_scale=log_scale, hide_errorbar=hide_errorbar)

    # Partial AUC below different max-FPR values
    for j, f in enumerate(FPR_MAX_PAUC):
        plot_dict = dict()
        plot_dict['x_label'] = x_label
        plot_dict['y_label'] = 'Partial AUROC (FPR <= {:.4f})'.format(f)
        plot_dict['title'] = "Partial area under ROC curve (FPR <= {:.4f})".format(f)
        for m in methods:
            d = results_dict[m]
            y_med = np.array([v[j] for v in d['pauc']['median']])
            y_low = np.array([v[j] for v in d['pauc']['CI_lower']])
            y_up = np.array([v[j] for v in d['pauc']['CI_upper']])
            plot_dict[m] = {
                'x_vals': 100 * d['proportion'],
                'y_vals': y_med,
                'y_low': y_low,
                'y_up': y_up,
                'y_err': [y_med - y_low, y_up - y_med]
            }

        if name_prefix:
            plot_file = os.path.join(output_dir, '{}_{}_{:d}'.format(name_prefix, 'pauc', j + 1))
        else:
            plot_file = os.path.join(output_dir, '{}_{:d}'.format('pauc', j + 1))

        plot_helper(plot_dict, methods, plot_file, min_yrange=0.1, place_legend_outside=place_legend_outside,
                    log_scale=log_scale, hide_errorbar=hide_errorbar)

    # TPR for different target FPR values
    for j, f in enumerate(FPR_THRESH):
        plot_dict = dict()
        plot_dict['x_label'] = x_label
        plot_dict['y_label'] = 'TPR at FPR = {:.4f}'.format(f)
        plot_dict['title'] = 'TPR at FPR = {:.4f}'.format(f)
        for m in methods:
            d = results_dict[m]
            y_med = np.array([v[j] for v in d['tpr']['median']])
            y_low = np.array([v[j] for v in d['tpr']['CI_lower']])
            y_up = np.array([v[j] for v in d['tpr']['CI_upper']])
            # Excess FPR above the target value `f`
            fpr_arr = np.clip([v[j] / f for v in d['fpr']['median']], 1., None)
            plot_dict[m] = {
                'x_vals': 100 * d['proportion'],
                'y_vals': y_med,
                'y_low': y_low,
                'y_up': y_up,
                'y_err': [y_med - y_low, y_up - y_med]
            }

        if name_prefix:
            plot_file = os.path.join(output_dir, '{}_{}_{:d}'.format(name_prefix, 'tpr', j + 1))
        else:
            plot_file = os.path.join(output_dir, '{}_{:d}'.format('tpr', j + 1))

        plot_helper(plot_dict, methods, plot_file, min_yrange=0.1, place_legend_outside=place_legend_outside,
                    log_scale=log_scale, hide_errorbar=hide_errorbar)


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
