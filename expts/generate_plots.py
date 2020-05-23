# Generate plots of the performance metrics from the metrics pickle files
import os
import pickle
import argparse
from helpers.utils import plot_performance_comparison


FILE_PREFIX = 'detection_metrics_'
FILE_EXT = '.pkl'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', '-o', required=True, help='output directory with the performance '
                                                                  'metrics files')
    parser.add_argument('--plot-dir', '-p', default='', help='directory to save the plot files')
    parser.add_argument('--pos-label', '--pl', default='adversarial',
                        help='label for the positive class - e.g. adversarial or ood')
    parser.add_argument('--name-prefix', '--pre', default='', help='optional prefix for the plot filenames')
    parser.add_argument('--log-scale', action='store_true', default=False,
                        help='use log scale for the x-axis of the plots')
    args = parser.parse_args()

    methods = []
    filenames = []
    lp = len(FILE_PREFIX)
    for f in os.listdir(args.output_dir):
        if f.startswith(FILE_PREFIX) and f.endswith(FILE_EXT):
            tmp = os.path.splitext(f)[0]
            methods.append(tmp[lp:])
            filenames.append(os.path.join(args.output_dir, f))

    results = dict()
    for m, fname in zip(methods, filenames):
        with open(fname, 'rb') as fp:
            results[m] = pickle.load(fp)

    if args.plot_dir:
        plot_dir = args.plot_dir
    else:
        plot_dir = args.output_dir

    # legend inside or outside the plot
    if len(methods) <= 8:
        plo = False
    else:
        plo = True

    if results:
        plot_performance_comparison(results, plot_dir, place_legend_outside=plo, pos_label=args.pos_label,
                                    log_scale=args.log_scale, hide_errorbar=True, name_prefix=args.name_prefix)
    else:
        print("No performance metrics files were found in the specified output directory.")


if __name__ == '__main__':
    main()
