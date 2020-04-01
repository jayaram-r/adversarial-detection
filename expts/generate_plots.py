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
    parser.add_argument('--pos-label', '-p', default='adversarial',
                        help='label for the positive class - e.g. adversarial or ood')
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

    if results:
        plot_performance_comparison(results, args.output_dir, place_legend_outside=True, pos_label=args.pos_label)
    else:
        print("No performance metrics files were found in the specified output directory.")


if __name__ == '__main__':
    main()
