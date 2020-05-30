#!/bin/bash
x_axis='proportion'
pos_label='OOD'
plot_dir="/nobackup/varun/adversarial-detection/expts/plots_all/${pos_label}/$x_axis"
plot_dir_sel="/nobackup/varun/adversarial-detection/expts/plots/${pos_label}/$x_axis"

for model in 'cifar10' 'mnist'; do
    base_dir="/nobackup/varun/adversarial-detection/expts/outputs/${model}/outliers"
    out_dir="${base_dir}/all/$x_axis"
    if [ $model == 'mnist' ]; then
        pre='mnist_notmnist'
    fi
    if [ $model == 'cifar10' ]; then
        pre='cifar10_svhn'
    fi
    if [ $model == 'svhn' ]; then
        pre='svhn_cifar10'
    fi
    
    python -u generate_plots.py -o $out_dir -p $plot_dir --pos-label $pos_label --x-axis $x_axis --pre $pre --hide-legend
done

for ext in 'png' 'pdf'; do
    if [ -d $plot_dir_sel/$ext ]; then
        rm -rf $plot_dir_sel/$ext
    fi
    mkdir -p $plot_dir_sel/$ext

    cp $plot_dir/*avg_prec*.$ext $plot_dir_sel/$ext/
    cp $plot_dir/*auc.$ext $plot_dir_sel/$ext/
    cp $plot_dir/*pauc_2.$ext $plot_dir_sel/$ext/
    cp $plot_dir/*tpr_5.$ext $plot_dir_sel/$ext/
done
