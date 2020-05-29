#!/bin/bash
gpu='3'
x_var='norm'
#x_var='proportion'

for model in 'svhn'; do
    base_dir="/nobackup/varun/adversarial-detection/expts/outputs/${model}/detection"
    #for attack in 'CW' 'PGD' 'FGSM' 'Custom'; do
    for attack in 'CW' 'Custom'; do
        echo "Calculating performance metrics on ${model}, ${attack}"

        target_dir="${base_dir}/${attack}/all/$x_var"
        if [ ! -d $target_dir ]; then
            mkdir -p $target_dir
        fi
        #proposed method
        output_dir="${base_dir}/${attack}/proposed"
        for ts in 'multinomial' 'trust'; do
            python -u calculate_performance.py -o $output_dir --x-var $x_var -m $model --dm proposed --ts $ts --st pvalue --pf fisher --adv-attack $attack --gpu $gpu
            python -u calculate_performance.py -o $output_dir --x-var $x_var -m $model --dm proposed --ts $ts --st pvalue --pf harmonic_mean --adv-attack $attack --gpu $gpu
            python -u calculate_performance.py -o $output_dir --x-var $x_var -m $model --dm proposed --ts $ts --st klpe --adv-attack $attack --gpu $gpu
        done
        cp ${output_dir}/${x_var}/detection_metrics_propo_multi_pval_fis_adv.pkl ${target_dir}/
        cp ${output_dir}/${x_var}/detection_metrics_propo_multi_klpe_adv.pkl ${target_dir}/

        #deep knn
        output_dir="${base_dir}/${attack}/deep_knn_no_dr"
        python -u calculate_performance.py -o $output_dir --x-var $x_var -m $model --dm dknn --adv-attack $attack --gpu $gpu
        cp ${output_dir}/${x_var}/detection_metrics_deep_KNN.pkl ${target_dir}/

        #LID
        output_dir="${base_dir}/${attack}/LID"
        python -u calculate_performance.py -o $output_dir --x-var $x_var -m $model --dm lid --adv-attack $attack --gpu $gpu
        cp ${output_dir}/${x_var}/detection_metrics_LID_ICLR.pkl ${target_dir}/

        #Mahalanobis
        output_dir="${base_dir}/${attack}/mahalanobis"
        python -u calculate_performance.py -o $output_dir --x-var $x_var -m $model --dm mahalanobis --adv-attack $attack --gpu $gpu
        cp ${output_dir}/${x_var}/detection_metrics_deep_mahalanobis.pkl ${target_dir}/

        #odds are odd
        output_dir="${base_dir}/${attack}/odds"
        python -u calculate_performance.py -o $output_dir --x-var $x_var -m $model --dm odds --adv-attack $attack --gpu $gpu
        cp ${output_dir}/${x_var}/detection_metrics_odds_are_odd.pkl ${target_dir}/

        #trust score
        output_dir="${base_dir}/${attack}/trust_score"
        python -u calculate_performance.py -o $output_dir --x-var $x_var -m $model --dm trust --lts prelogit --adv-attack $attack --gpu $gpu
        cp ${output_dir}/${x_var}/detection_metrics_trust_prelogit.pkl ${target_dir}/
    done
done
