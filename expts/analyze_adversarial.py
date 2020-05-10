"""
Analyze the norm difference of adversarial samples.
USAGE:
python analyze_adversarial.py -d <path with numpy data files> -n <norm type> -o <output directory>

"""
import argparse
import numpy as np
import os
from helpers.utils import check_label_mismatch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze_adversarial_samples(data_path, output_path, norm_type='inf', train_only=False, test_only=False):
    if train_only and test_only:
        raise ValueError("Both inputs 'train_only' and 'test_only' cannot be set to True")

    norm_diff_list = []
    if not test_only:
        # Perturbed inputs
        data_tr_adv = np.load(os.path.join(data_path, "data_tr_adv.npy"))
        # Clean inputs
        data_tr_clean = np.load(os.path.join(data_path, "data_tr_clean.npy"))
        # Predicted (mis-classified) labels
        labels_pred_tr = np.load(os.path.join(data_path, "labels_tr_adv.npy"))
        # Labels of the original inputs from which the adversarial inputs were created
        labels_tr = np.load(os.path.join(data_path, "labels_tr_clean.npy"))

        # Flatten all but the first dimension and take the vector norm of each row in `diff`
        n_train = data_tr_clean.shape[0]
        diff = data_tr_adv.reshape(n_train, -1) - data_tr_clean.reshape(n_train, -1)
        if norm_type == 'inf':
            norm_diff_tr = np.linalg.norm(diff, ord=np.inf, axis=1)
        else:
            # expecting a non-negative integer
            norm_diff_tr = np.linalg.norm(diff, ord=int(norm_type), axis=1)

        mask_adv = norm_diff_tr > 1e-16
        norm_diff_list.append(norm_diff_tr[mask_adv])
        # Check if the original and adversarial labels are all different
        check_label_mismatch(labels_tr[mask_adv], labels_pred_tr[mask_adv])
        print("Train fold: number of samples = {:d}, number of adversarial samples = {:d}".
              format(mask_adv.shape[0], mask_adv[mask_adv].shape[0]))

    if not train_only:
        # Perturbed inputs
        data_te_adv = np.load(os.path.join(data_path, "data_te_adv.npy"))
        # Clean inputs
        data_te_clean = np.load(os.path.join(data_path, "data_te_clean.npy"))
        # Predicted (mis-classified) labels
        labels_pred_te = np.load(os.path.join(data_path, "labels_te_adv.npy"))
        # Labels of the original inputs from which the adversarial inputs were created
        labels_te = np.load(os.path.join(data_path, "labels_te_clean.npy"))

        # Flatten all but the first dimension and take the vector norm of each row in `diff`
        n_test = data_te_clean.shape[0]
        diff = data_te_adv.reshape(n_test, -1) - data_te_clean.reshape(n_test, -1)
        if norm_type == 'inf':
            norm_diff_te = np.linalg.norm(diff, ord=np.inf, axis=1)
        else:
            # expecting a non-negative integer
            norm_diff_te = np.linalg.norm(diff, ord=int(norm_type), axis=1)

        mask_adv = norm_diff_te > 1e-16
        norm_diff_list.append(norm_diff_te[mask_adv])
        # Check if the original and adversarial labels are all different
        check_label_mismatch(labels_te[mask_adv], labels_pred_te[mask_adv])
        print("Test fold: number of samples = {:d}, number of adversarial samples = {:d}".
              format(mask_adv.shape[0], mask_adv[mask_adv].shape[0]))

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    norm_diff = np.concatenate(norm_diff_list)
    perc = [0, 1, 25, 50, 75, 99, 100]
    lines = ['{}\t{}\n'.format('perc', 'value')]
    v = np.percentile(norm_diff, perc)
    print("\nNumber of adversarial samples = {:d}".format(norm_diff.shape[0]))
    print("Percentile of {}-norm values:".format(norm_type))
    for a, b in zip(perc, v):
        print("{:2d}\t{:.8f}".format(a, b))
        lines.append("{:2d}\t{:.8f}\n".format(a, b))

    fname = os.path.join(output_path, '{}_norm_percentiles.csv'.format(norm_type))
    with open(fname, 'w') as fp:
        fp.writelines(lines)

    # Histogram of the norm values
    fig = plt.figure()
    m = norm_diff.shape[0]
    if m < 40:
        n_bins = 2
    elif m < 1000:
        n_bins = int(np.ceil(m / 20.))
    else:
        n_bins = 50

    _ = plt.hist(norm_diff, n_bins, density=True, facecolor='blue', alpha=0.5)
    plt.xlabel('perturbation norm')
    plt.title('Normalized histogram of perturbation norm (L-{})'.format(norm_type))
    fname = os.path.join(output_path, '{}_norm_histogram.png'.format(norm_type))
    fig.savefig(fname, dpi=600, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', '-d', required=True, help='directory with the numpy data files for the '
                                                                 'adversarial attack')
    parser.add_argument('--norm', '-n', default='inf', help="Norm type to use. For example 'inf', '1', '2', '0'")
    parser.add_argument('--output-dir', '-o', default='', help='directory for output files')
    parser.add_argument('--test-only', action='store_true', default=False,
                        help='Use option to analyze only the test fold data')
    parser.add_argument('--train-only', action='store_true', default=False,
                        help='Use option to analyze only the train fold data')
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), 'norm_analysis')

    analyze_adversarial_samples(args.data_path, output_dir, norm_type=args.norm, train_only=args.train_only,
                                test_only=args.test_only)


if __name__ == '__main__':
    main()
