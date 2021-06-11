"""
Simple utility to generate a figure with legends.
https://stackoverflow.com/questions/4534480/get-legend-as-a-separate-picture-in-matplotlib
https://stackoverflow.com/questions/50268254/matplotlib-convert-the-legend-to-a-bitmap

"""
import os
import sys
import argparse
import numpy as np
from helpers.constants import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', '-o', required=True, help='output directory')
    args = parser.parse_args()
    # Adversarial
    methods = [
        'LID', 'Deep KNN', 'Deep Mahalanobis', 'Odds are ood', '{}, LPE, multi'.format(METHOD_NAME_PROPOSED),
        '{}, Fisher, multi'.format(METHOD_NAME_PROPOSED), 'Trust Score'
    ]
    n_col = 4
    '''
    # OOD
    methods = [
        'Deep KNN', 'Deep Mahalanobis', '{}, LPE, multi'.format(METHOD_NAME_PROPOSED),
        '{}, Fisher, multi'.format(METHOD_NAME_PROPOSED), 'Trust Score'
    ]
    n_col = 3
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for j, m in enumerate(methods):
        ax.plot(np.arange(10), np.arange(10),
                linestyle='-', linewidth=0.75, color=COLORS[j], marker=MARKERS[j], label=m)

    _ = ax.legend(
        loc='center', prop={'size': 'medium', 'weight': 'normal'}, frameon=False, ncol=4
        #, fancybox=True, framealpha=0.7
    )

    # Legend figure. Adjust the figure size as necessary
    fig_leg = plt.figure(figsize=(3, 3))
    ax_leg = fig_leg.add_subplot(111)
    # Add the legend from the previous figure
    # Legend font sizes: xx-small, x-small, small, medium, large, x-large, xx-large
    ax_leg.legend(
        *ax.get_legend_handles_labels(), loc='center', prop={'size': 'medium', 'weight': 'normal'},
        frameon=False, ncol=n_col
        #, fancybox=True, framealpha=0.7
    )
    # hide the axes frame and the x/y labels
    ax_leg.axis('off')
    fig_leg.savefig('legend.png')

    plot_file = os.path.join(args.output_dir, 'legend')
    # fig_leg.tight_layout()
    fig_leg.savefig('{}.png'.format(plot_file), dpi=600, bbox_inches='tight')
    fig_leg.savefig('{}.pdf'.format(plot_file), dpi=600, bbox_inches='tight')
    plt.close(fig)
    plt.close(fig_leg)


if __name__ == '__main__':
    main()
