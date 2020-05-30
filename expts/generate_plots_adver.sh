#!/bin/bash
x_axis='norm'
#x_axis='proportion'
pos_label='adversarial'
plot_dir="/nobackup/varun/adversarial-detection/expts/plots_all/${pos_label}/$x_axis"
plot_dir_sel="/nobackup/varun/adversarial-detection/expts/plots/${pos_label}/$x_axis"

for model in 'svhn' 'cifar10' 'mnist'; do
    base_dir="/nobackup/varun/adversarial-detection/expts/outputs/${model}/detection"
    for attack in 'CW' 'PGD' 'FGSM' 'Custom'; do
        echo "Plotting ${model}, ${attack}"
        out_dir="${base_dir}/${attack}/all/$x_axis"
        python -u generate_plots.py -o $out_dir -p $plot_dir --pos-label $pos_label --x-axis $x_axis --pre ${model}_${attack} --hide-legend
    done
done

for ext in 'png' 'pdf'; do
    if [ -d $plot_dir_sel/$ext ]; then
        rm -rf $plot_dir_sel/$ext
    fi
    mkdir -p $plot_dir_sel/$ext

    cp $plot_dir/*avg_prec*.$ext $plot_dir_sel/$ext/
    cp $plot_dir/*auc.$ext $plot_dir_sel/$ext/
    cp $plot_dir/*pauc_4.$ext $plot_dir_sel/$ext/
    cp $plot_dir/*tpr_5.$ext $plot_dir_sel/$ext/
done
