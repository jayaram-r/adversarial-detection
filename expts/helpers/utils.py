
import numpy as np
import torch
import os
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
    SEED_DEFAULT,
    FPR_MAX_PAUC,
    FPR_THRESH,
    COLORS
)
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


def get_data_bounds(data, alpha=0.95):
    bounds = [np.min(data), np.max(data)]
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


def metrics_varying_positive_class_proportion(scores, labels, pos_label=1, num_prop=10,
                                              num_random_samples=100, n_jobs=1, seed=SEED_DEFAULT, output_file=None):
    """
    Calculate a number of performance metrics as the fraction of positive samples in the data is varied.
    For each proportion, the estimates are calculated from different random samples, and the median and confidence
    interval values of each performance metric are reported.

    :param scores: numpy array with the detection scores.
    :param labels: numpy array with the binary detection labels.
    :param pos_label: postive class label; set to 1 by default.
    :param num_prop: number of positive proportion values to evaluate.
    :param num_random_samples: number of random samples to use for estimating the median and confidence interval.
    :param n_jobs: number of cpu cores or processes to use in parallel.
    :param seed: seed for the random number generator.
    :param output_file: (optional) path to an output file where the metrics dict is written to using the
                        Pickle protocol.

    :return: a dict with the proportion of positive samples and all the performance metrics.
    """
    np.random.seed(seed)
    n_folds = len(scores)
    n_samp = []
    ind_pos = []
    n_pos_max = []
    ind_neg = []
    scores_neg = []
    labels_neg = []
    for i in range(n_folds):
        n_samp.append(float(labels[i].shape[0]))
        # index of positive labels
        temp = np.where(labels[i] == pos_label)[0]
        ind_pos.append(temp)
        n_pos_max.append(temp.shape[0])
        # index of negative labels
        temp = np.where(labels[i] != pos_label)[0]
        ind_neg.append(temp)
        scores_neg.append(scores[i][temp])
        labels_neg.append(labels[i][temp])

    # Minimum proportion of positive samples. Ensuring that there is at least 1 positive sample
    p_min = max([max(1., np.ceil(0.005 * n_samp[i])) / n_samp[i] for i in range(n_folds)])
    # Maximum proportion of positive samples considering all the folds
    p_max = min([n_pos_max[i] / n_samp[i] for i in range(n_folds)])
    # Range of proportion of positive samples
    prop_range = np.unique(np.linspace(p_min, p_max, num=num_prop))

    # dict with the positive proportion and the corresponding performance metrics. The mean, and lower and upper
    # confidence interval values are calculated for each performance metric
    results = {
        'proportion': prop_range.tolist(),
        'auc': {'median': [], 'CI_lower': [], 'CI_upper': []},
        'avg_prec': {'median': [], 'CI_lower': [], 'CI_upper': []},
        'pauc': {'median': [], 'CI_lower': [], 'CI_upper': []},
        'tpr': {'median': [], 'CI_lower': [], 'CI_upper': []},
        'fpr': {'median': [], 'CI_lower': [], 'CI_upper': []}
    }
    metric_names = results.keys()
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
        print("\nCalculating performance metrics for target positive proportion: {:.4f}".format(p))

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

            print("Fold {:d}. Number of positive samples = {:d}. Target proportion = {:.4f}. Actual proportion"
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
        print("TPR\tFPR")
        for a, b in zip(ret1[1], ret2[1]):
            print("{:.6f}\t{:.6f}".format(a, b))

    # Save the results to a pickle file if required
    if output_file:
        with open(output_file, 'wb') as fp:
            pickle.dump(results, fp)

    return results


def plot_helper(plot_dict, methods, plot_file):
    fig = plt.figure()
    j = 0
    x_vals = []
    y_vals = []
    for m in methods:
        d = plot_dict[m]
        plt.plot(d['x_vals'], d['y_vals'], linestyle='--', color=COLORS[j], marker='.', label=m)
        x_vals.extend(d['x_vals'])
        y_vals.extend(d['y_vals'])
        j += 1

    x_bounds = get_data_bounds(x_vals, alpha=0.99)
    y_bounds = get_data_bounds(y_vals, alpha=0.99)
    plt.xlim([x_bounds[0], x_bounds[1]])
    plt.ylim([y_bounds[0], y_bounds[1]])
    plt.xticks(np.linspace(x_bounds[0], x_bounds[1], num=6))
    plt.yticks(np.linspace(y_bounds[0], y_bounds[1], num=6))

    plt.xlabel(plot_dict['x_label'], fontsize=10, fontweight='bold')
    plt.ylabel(plot_dict['y_label'], fontsize=10, fontweight='bold')
    plt.title(plot_dict['title'], fontsize=10, fontweight='bold')
    plt.legend(loc='best')
    plt.plot()
    fig.savefig(plot_file, dpi=600, bbox_inches='tight')


def plot_performance_comparison(results_dict, output_dir):
    """
    Plot the performance comparison for different detection methods.

    :param results_dict: dict mapping each method name to its metrics dict (obtained from the function
                         `metrics_varying_positive_class_proportion`).
    :param output_dir: path to the output directory where the plots are to be saved.
    :return: None
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    x_label = 'percentage of positive samples'
    methods = sorted(results_dict.keys())
    # AUC plots
    plot_dict = dict()
    plot_dict['x_label'] = x_label
    plot_dict['y_label'] = 'AUC'
    plot_dict['title'] = 'Area under the ROC curve'
    for m in methods:
        d = results_dict[m]
        plot_dict[m] = {
            'x_vals': 100 * d['proportion'],
            'y_vals': d['auc']['median']
        }

    plot_file = os.path.join(output_dir, '{}_comparison.png'.format('auc'))
    plot_helper(plot_dict, methods, plot_file)

    # Average precision plots
    plot_dict = dict()
    plot_dict['x_label'] = x_label
    plot_dict['y_label'] = 'Avg. precision'
    plot_dict['title'] = 'Average precision or area under the PR curve'
    for m in methods:
        d = results_dict[m]
        plot_dict[m] = {
            'x_vals': 100 * d['proportion'],
            'y_vals': d['avg_prec']['median']
        }

    plot_file = os.path.join(output_dir, '{}_comparison.png'.format('avg_prec'))
    plot_helper(plot_dict, methods, plot_file)

    # Partial AUC for different max-FPR values
    for j, f in enumerate(FPR_MAX_PAUC):
        plot_dict = dict()
        plot_dict['x_label'] = x_label
        plot_dict['y_label'] = 'p-AUC'
        plot_dict['title'] = "Partial area under the ROC curve (FPR <= {:.4f})".format(f)
        for m in methods:
            d = results_dict[m]
            y_vals = [v[j] for v in d['pauc']['median']]
            plot_dict[m] = {
                'x_vals': 100 * d['proportion'],
                'y_vals': y_vals
            }

        plot_file = os.path.join(output_dir, '{}_comparison_{:d}.png'.format('pauc', j + 1))
        plot_helper(plot_dict, methods, plot_file)


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
