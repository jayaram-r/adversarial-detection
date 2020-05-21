"""
Deep Mahalanobis detection method repurposed from the author's implementation:
https://github.com/pokaxpoka/deep_Mahalanobis_detector

"""
from nets.svhn import *
from nets.cifar10 import *
from nets.mnist import *
from nets.resnet import *
from helpers.utils import (
    get_samples_as_ndarray,
    convert_to_loader
)
import detectors.deep_mahalanobis.lib_generation as lib_generation
import detectors.deep_mahalanobis.lib_regression as lib_regression
import torch
import numpy as np
import os
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import MinMaxScaler


# Noise magnitude values
NOISE_MAG_LIST = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]


def get_mahalanobis_labels(model, device, data_te_clean, data_te_noisy, data_te_adv, labels_te_clean):
    # Convert numpy arrays to torch data loaders with appropriate dtype and device
    data_loader = convert_to_loader(data_te_clean, labels_te_clean, dtype_x=torch.float, device=device)
    noisy_data_loader = convert_to_loader(data_te_noisy, labels_te_clean, dtype_x=torch.float, device=device)
    adv_data_loader = convert_to_loader(data_te_adv, labels_te_clean, dtype_x=torch.float, device=device)

    total = 0
    selected_list = []
    selected_index = 0
    with torch.no_grad():
        for elements in zip(adv_data_loader, noisy_data_loader, data_loader):
            adv_data, _ = elements[0]
            # adv_data = adv_data.to(device=device, dtype=torch.float)
            noisy_data, _ = elements[1]
            # noisy_data = noisy_data.to(device=device, dtype=torch.float)
            data, target = elements[2]
            # data = data.to(device=device, dtype=torch.float)
            # target = target.to(device=device)
            n_batch = data.size(0)

            # predictions on adversarial data
            output_adv = model(adv_data)
            pred_adv = output_adv.max(1)[1]
            equal_flag_adv = pred_adv.eq(target).cpu()

            # predictions on noisy data
            output_noisy = model(noisy_data)
            pred_noise = output_noisy.max(1)[1]
            equal_flag_noise = pred_noise.eq(target).cpu()

            # predictions on clean data
            output = model(data)
            pred = output.max(1)[1]
            equal_flag = pred.eq(target).cpu()

            if total == 0:
                label_tot = target.clone().cpu()
            else:
                label_tot = torch.cat((label_tot, target.clone().cpu()), 0)

            for i in range(n_batch):
                if equal_flag[i] == 1 and equal_flag_noise[i] == 1 and equal_flag_adv[i] == 0:
                    # Correct prediction on the clean and noisy sample, but incorrect prediction on the
                    # adversarial sample
                    selected_list.append(selected_index)

                selected_index += 1

            total += n_batch

    # return labels of the selected indices as a numpy array
    selected_list = torch.LongTensor(selected_list)
    label_tot = torch.index_select(label_tot, 0, selected_list)
    return label_tot.detach().cpu().numpy()


def calc_mahalanobis_features(model, device, net_type, num_labels, data_tr, num_layers,
                              sample_mean, precision, noise_mag):
    mahalanobis_feat = None
    for i in range(num_layers):
        M_in = lib_generation.get_Mahalanobis_score_adv(
            model, device, data_tr, num_labels, net_type, sample_mean, precision, i, noise_mag
        )
        M_in = np.asarray(M_in, dtype=np.float32)
        if i == 0:
            mahalanobis_feat = M_in.reshape((M_in.shape[0], -1))
        else:
            mahalanobis_feat = np.concatenate((mahalanobis_feat, M_in.reshape((M_in.shape[0], -1))), axis=1)

    # output array has shape `(n_samples, n_layers)`
    return np.asarray(mahalanobis_feat, dtype=np.float32)


def fit_mahalanobis_scores(model, device, adv_type, net_type, num_labels, outf, train_loader, data_tr_clean,
                           data_tr_adv, data_tr_noisy, n_jobs=-1):
    # numpy arrays to torch tensors
    data_tr_clean = torch.from_numpy(data_tr_clean).to(device=device, dtype=torch.float)
    data_tr_adv = torch.from_numpy(data_tr_adv).to(device=device, dtype=torch.float)
    data_tr_noisy = torch.from_numpy(data_tr_noisy).to(device=device, dtype=torch.float)

    if model.training:
        model.eval()

    # Extract the layer-wise embeddings for a random input with 2 samples
    size_data = data_tr_clean.size()
    temp_x = torch.rand(2, size_data[1], size_data[2], size_data[3]).to(device)
    _, temp_list = model.layer_wise_deep_mahalanobis(temp_x)
    num_output = len(temp_list)
    print("Number of layer embeddings: {:d}".format(num_output))
    
    feature_list = np.zeros(num_output, dtype=np.int)
    for i, out in enumerate(temp_list):
        feature_list[i] = out.size(1)   # num. channels for conv. layers; num. dimensions for FC layers
        
    print('Calculating the sample mean and covariance matrix')
    sample_mean, precision = lib_generation.sample_estimator(model, device, num_labels, feature_list, train_loader)

    # Noise magnitude values
    noise_mag_list = NOISE_MAG_LIST
    model_dict_best = {}
    noise_mag_best = NOISE_MAG_LIST[0]
    auc_max = -1.
    for magnitude in noise_mag_list:
        print('\nNoise: ' + str(magnitude))
        print('Calculating the Mahalanobis score (features) from the layers')
        # Clean/in-distribution data
        Mahalanobis_in = calc_mahalanobis_features(model, device, net_type, num_labels, data_tr_clean, num_output,
                                                   sample_mean, precision, magnitude)
        # OOD/adversarial data
        Mahalanobis_out = calc_mahalanobis_features(model, device, net_type, num_labels, data_tr_adv, num_output,
                                                    sample_mean, precision, magnitude)
        # Noisy data
        Mahalanobis_noisy = calc_mahalanobis_features(model, device, net_type, num_labels, data_tr_noisy, num_output,
                                                      sample_mean, precision, magnitude)
        # arrays have shape `(n_samples, n_layers)`
        Mahalanobis_pos = np.concatenate((Mahalanobis_in, Mahalanobis_noisy))
        Mahalanobis_feat, Mahalanobis_labels = lib_generation.merge_and_generate_labels(Mahalanobis_out,
                                                                                        Mahalanobis_pos)
        file_name = os.path.join(outf, 'Mahalanobis_%s_%s_%s.npy' % (str(magnitude), net_type, adv_type))
        
        mahalanobis_data = np.concatenate((Mahalanobis_feat, Mahalanobis_labels), axis=1)
        np.save(file_name, mahalanobis_data)

        print("Training a logistic classifier to discriminate in-distribution from OOD/adversarial samples")
        model_dict_curr = train_logistic_classifier(mahalanobis_data, scale_features=False, balance_classes=False,
                                                    n_jobs=n_jobs)
        auc_curr = model_dict_curr['auc_avg']
        if auc_curr > auc_max:
            auc_max = auc_curr
            model_dict_best = model_dict_curr
            noise_mag_best = magnitude

    print("\nNoise magnitude {:.6f} resulted in the logistic classifier with maximum average AUC = {:.6f}".
          format(noise_mag_best, auc_max))
    model_detector = model_dict_best
    model_detector['sample_mean'] = sample_mean
    model_detector['precision'] = precision
    model_detector['noise_magnitude'] = noise_mag_best
    model_detector['n_classes'] = num_labels
    model_detector['n_layers'] = num_output

    return model_detector


def train_logistic_classifier(mahalanobis_data, n_cv_folds=5,
                              scale_features=False, balance_classes=False,
                              n_jobs=-1, max_iter=200, seed_rng=123):
    features = mahalanobis_data[:, :-1]
    labels = mahalanobis_data[:, -1].astype(np.int)     # binary 0/1 labels

    lab_uniq, count_classes = np.unique(labels, return_counts=True)
    if len(lab_uniq) == 2 and lab_uniq[0] == 0 and lab_uniq[1] == 1:
        pass
    else:
        raise ValueError("Did not receive expected binary class labels 0 and 1.")

    pos_prop = float(count_classes[1]) / (count_classes[0] + count_classes[1])
    if scale_features:
        # Min-max scaling to preprocess all features to the same range [0, 1]
        scaler = MinMaxScaler().fit(features)
        features = scaler.transform(features)
    else:
        scaler = None

    print("\nTraining a binary logistic classifier with {:d} samples and {:d} Mahalanobis features.".
          format(*features.shape))
    print("Using {:d}-fold cross-validation with area under ROC curve as the metric to select the best "
          "regularization hyperparameter.".format(n_cv_folds))
    print("Proportion of positive (adversarial or OOD) samples in the training data: {:.4f}".format(pos_prop))
    if pos_prop <= 0.1:
        # high imbalance in the classes
        balance_classes = True

    class_weight = None
    if balance_classes:
        if (pos_prop < 0.45) or (pos_prop > 0.55):
            class_weight = {0: 1.0 / (1 - pos_prop),
                            1: 1.0 / pos_prop}
            print("Balancing the classes by assigning sample weight {:.4f} to class 0 and sample weight {:.4f} "
                  "to class 1".format(class_weight[0], class_weight[1]))

    model_logistic = LogisticRegressionCV(
        cv=n_cv_folds,
        penalty='l2',
        scoring='roc_auc',
        multi_class='auto',
        class_weight=class_weight,
        max_iter=max_iter,
        refit=True,
        n_jobs=n_jobs,
        random_state=seed_rng
    ).fit(features, labels)

    # regularization coefficient values
    coeffs = model_logistic.Cs_
    # regularization coefficient corresponding to the maximum cross-validated AUC
    coeff_best = model_logistic.C_[0]
    mask = np.abs(coeffs - coeff_best) < 1e-16
    ind = np.where(mask)[0][0]
    # average AUC across the test folds for the best regularization coefficient
    auc_scores = model_logistic.scores_[1]      # has shape `(n_cv_folds, coeffs.shape[0])`
    auc_avg_best = np.mean(auc_scores[:, ind])
    print("Average AUC from the test folds: {:.6f}".format(auc_avg_best))

    # proba = model_logistic.predict_proba(features)[:, -1]
    model_dict = {'logistic': model_logistic,
                  'scaler': scaler,
                  'auc_avg': auc_avg_best}
    return model_dict


def get_mahalanobis_scores(model_detector, data_te, model_dnn, device, net_type):
    # numpy array to torch tensors
    data_te = torch.from_numpy(data_te).to(device=device, dtype=torch.float)

    print('Calculating the Mahalanobis score (features) from the layers')
    features_te = calc_mahalanobis_features(
        model_dnn, device, net_type, model_detector['n_classes'], data_te, model_detector['n_layers'],
        model_detector['sample_mean'], model_detector['precision'], model_detector['noise_magnitude']
    )
    if model_detector['scaler']:
        # Min-max scaling to preprocess all features to the same range [0, 1]
        features_te = model_detector['scaler'].transform(features_te)

    # probability of the positive (ood or adversarial) class
    return model_detector['logistic'].predict_proba(features_te)[:, -1]


# Not used
def get_mahalanobis_scores_alt(adv_type, dataset, outf):
    #train_loader, test_clean_data, test_adv_data, test_noisy_data, test_label):

    # initial setup
    # dataset_list = ['cifar10', 'cifar100', 'svhn']
    # adv_test_list = ['FGSM', 'BIM', 'DeepFool', 'CWL2']
    dataset_list = [dataset]
    adv_test_list = [adv_type]
    print('evaluate the Mahalanobis estimator')
    score_list = ['Mahalanobis_0.0', 'Mahalanobis_0.01', 'Mahalanobis_0.005',
                  'Mahalanobis_0.002', 'Mahalanobis_0.0014', 'Mahalanobis_0.001', 'Mahalanobis_0.0005']
    list_best_results_ours, list_best_results_index_ours = [], []
    for dataset in dataset_list:
        print('load train data: ', dataset)
        list_best_results_out, list_best_results_index_out = [], []
        for out_type in adv_test_list:
            best_auroc, best_result, best_index = 0, 0, 0
            for score in score_list:
                print('load train data: ', out_type, ' of ', score)
                total_X, total_Y = lib_regression.load_characteristics(score, dataset, out_type, outf)
                # validation / test split
                X_val, Y_val, X_test, Y_test = lib_regression.block_split_adv(total_X, total_Y)
                pivot = int(X_val.shape[0] / 6)
                X_train = np.concatenate((X_val[:pivot], X_val[2*pivot:3*pivot], X_val[4*pivot:5*pivot]))
                Y_train = np.concatenate((Y_val[:pivot], Y_val[2*pivot:3*pivot], Y_val[4*pivot:5*pivot]))
                X_val_for_test = np.concatenate((X_val[pivot:2*pivot], X_val[3*pivot:4*pivot], X_val[5*pivot:]))
                Y_val_for_test = np.concatenate((Y_val[pivot:2*pivot], Y_val[3*pivot:4*pivot], Y_val[5*pivot:]))

                # Train a logistic classifier with cross-validation on the training split
                lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train)
                y_pred = lr.predict_proba(X_train)[:, 1]
                # print('training mse: {:.4f}'.format(np.mean(y_pred - Y_train)))

                y_pred = lr.predict_proba(X_val_for_test)[:, 1]
                # print('test mse: {:.4f}'.format(np.mean(y_pred - Y_val_for_test)))

                # performance on the test split
                results = lib_regression.detection_performance(lr, X_val_for_test, Y_val_for_test, outf)
                if best_auroc < results['TMP']['AUROC']:
                    # noise level corresponding to maximum AUROC
                    best_auroc = results['TMP']['AUROC']
                    best_index = score
                    best_result = lib_regression.detection_performance(lr, X_test, Y_test, outf)

            list_best_results_out.append(best_result)
            list_best_results_index_out.append(best_index)

        list_best_results_ours.append(list_best_results_out)
        list_best_results_index_ours.append(list_best_results_index_out)

    print("evaluation of Mahalanobis estimator (step 1)")
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    count_in = 0
    print("results of Mahalanobis")
    for in_list in list_best_results_ours:
        print('in_distribution: ' + dataset_list[count_in] + '==========')
        count_out = 0
        for results in in_list:
            print('out_distribution: ' + adv_test_list[count_out])
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')

            print('\n{val:6.2f}'.format(val=100.*results['TMP']['TNR']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
            print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')
            print('Input noise: ' + list_best_results_index_ours[count_in][count_out])
            print('')
            count_out += 1

        count_in += 1
