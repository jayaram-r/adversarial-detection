"""
Repurposed code from the Github repo associated with the paper:
Roth, Kevin, Yannic Kilcher, and Thomas Hofmann. "The odds are odd: A statistical test for detecting adversarial
examples." arXiv preprint arXiv:1902.04818 (2019).

Repo: https://github.com/yk/icml19_public
"""
from nets.svhn import *
from nets.cifar10 import *
from nets.mnist import *
from nets.resnet import *
from detectors.tf_robustify import *
import itertools as itt
from sklearn.metrics import confusion_matrix
from helpers.utils import get_samples_as_ndarray
import detectors.lib_generation as lib_generation
import detectors.lib_regression as lib_regression

import torch
import numpy as np
import os

from torch.autograd import Variable
from sklearn.linear_model import LogisticRegressionCV

def fit_mahalanobis_scores(model, adv_type, dataset, num_labels, outf,
        train_loader, test_clean_data, test_adv_data, test_noisy_data, test_label):
    
    outf = outf
    net_type = dataset

    test_clean_data = torch.from_numpy(test_clean_data)
    test_adv_data = torch.from_numpy(test_adv_data)
    test_noisy_data = torch.from_numpy(test_noisy_data)
    test_label = torch.from_numpy(test_label)

    # set information about feature extaction
    model.eval()
    
    if dataset == 'mnist':
        temp_x = torch.rand(2,1,28,28).cuda()
    elif dataset == 'cifar10':
        temp_x = torch.rand(2,3,32,32).cuda()
    elif dataset == 'svhn':
        temp_x = torch.rand(2,3,32,32).cuda()

    temp_x = Variable(temp_x)
    _, temp_list = model.layer_wise_deep_mahalanobis(temp_x)
    num_output = len(temp_list)
    
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        #out = out.view(2,-1)
        #print(out.shape, out.size(1))
        feature_list[count] = out.size(1)
        count += 1
        
    print('get sample mean and covariance')
    
    sample_mean, precision = lib_generation.sample_estimator(model, num_labels, feature_list, train_loader)

    
    print('get LID scores')
    LID, LID_adv, LID_noisy = lib_generation.get_LID(model, test_clean_data, test_adv_data, test_noisy_data, test_label, num_output)          
    
    overlap_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    list_counter = 0
    for overlap in overlap_list:
        Save_LID = np.asarray(LID[list_counter], dtype=np.float32)
        Save_LID_adv = np.asarray(LID_adv[list_counter], dtype=np.float32)
        Save_LID_noisy = np.asarray(LID_noisy[list_counter], dtype=np.float32)
        Save_LID_pos = np.concatenate((Save_LID, Save_LID_noisy))
        
        LID_data, LID_labels = lib_generation.merge_and_generate_labels(Save_LID_adv, Save_LID_pos)
        file_name = os.path.join(outf, 'LID_%s_%s_%s.npy' % (overlap, dataset, adv_type))
        LID_data = np.concatenate((LID_data, LID_labels), axis=1)
        np.save(file_name, LID_data)
        list_counter += 1
    
    print('get Mahalanobis scores')
    m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    for magnitude in m_list:
        print('\nNoise: ' + str(magnitude))
        
        for i in range(num_output):
            M_in = lib_generation.get_Mahalanobis_score_adv(model, test_clean_data, test_label, 
                    num_labels, outf, net_type, sample_mean, precision, i, magnitude)
            M_in = np.asarray(M_in, dtype=np.float32)
            if i == 0:
                Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
            else:
                Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)

        for i in range(num_output):
            M_out = lib_generation.get_Mahalanobis_score_adv(model, test_adv_data, test_label, 
                    num_labels, outf, net_type, sample_mean, precision, i, magnitude)
            M_out = np.asarray(M_out, dtype=np.float32)
            if i == 0:
                Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
            else:
                Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)
        
        for i in range(num_output):
            M_noisy = lib_generation.get_Mahalanobis_score_adv(model, test_noisy_data, test_label, 
                    num_labels, outf, net_type, sample_mean, precision, i, magnitude)
            M_noisy = np.asarray(M_noisy, dtype=np.float32)
            if i == 0:
                Mahalanobis_noisy = M_noisy.reshape((M_noisy.shape[0], -1))
            else:
                Mahalanobis_noisy = np.concatenate((Mahalanobis_noisy, M_noisy.reshape((M_noisy.shape[0], -1))), axis=1)            
        
        Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
        Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
        Mahalanobis_noisy = np.asarray(Mahalanobis_noisy, dtype=np.float32)
        Mahalanobis_pos = np.concatenate((Mahalanobis_in, Mahalanobis_noisy))

        Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_pos)
        file_name = os.path.join(outf, 'Mahalanobis_%s_%s_%s.npy' % (str(magnitude), dataset, adv_type))
        
        Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
        np.save(file_name, Mahalanobis_data)
        
    print('scores calculated')
        #return Mahalanobis_out

def get_mahalanobis_scores(model, adv_type, dataset, num_labels, outf):
        #train_loader, test_clean_data, test_adv_data, test_noisy_data, test_label):

    # initial setup
    #dataset_list = ['cifar10', 'cifar100', 'svhn']
    #adv_test_list = ['FGSM', 'BIM', 'DeepFool', 'CWL2']

    #Varun: do we care about the LID estimator? If not, it is commented out
    '''
    print('evaluate the LID estimator')
    score_list = ['LID_10', 'LID_20', 'LID_30', 'LID_40', 'LID_50', 'LID_60', 'LID_70', 'LID_80', 'LID_90']
    list_best_results, list_best_results_index = [], []
    for dataset in dataset_list:
        print('load train data: ', dataset)
        outf = './adv_output/' + args.net_type + '_' + dataset + '/'

        list_best_results_out, list_best_results_index_out = [], []
        for out in adv_test_list:
            best_auroc, best_result, best_index = 0, 0, 0
            for score in score_list:
                print('load train data: ', out, ' of ', score)
                total_X, total_Y = lib_regression.load_characteristics(score, dataset, out, outf)
                X_val, Y_val, X_test, Y_test = lib_regression.block_split_adv(total_X, total_Y)
                pivot = int(X_val.shape[0] / 6)
                X_train = np.concatenate((X_val[:pivot], X_val[2*pivot:3*pivot], X_val[4*pivot:5*pivot]))
                Y_train = np.concatenate((Y_val[:pivot], Y_val[2*pivot:3*pivot], Y_val[4*pivot:5*pivot]))
                X_val_for_test = np.concatenate((X_val[pivot:2*pivot], X_val[3*pivot:4*pivot], X_val[5*pivot:]))
                Y_val_for_test = np.concatenate((Y_val[pivot:2*pivot], Y_val[3*pivot:4*pivot], Y_val[5*pivot:]))
                lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train)
                y_pred = lr.predict_proba(X_train)[:, 1]
                #print('training mse: {:.4f}'.format(np.mean(y_pred - Y_train)))
                y_pred = lr.predict_proba(X_val_for_test)[:, 1]
                #print('test mse: {:.4f}'.format(np.mean(y_pred - Y_val_for_test)))
                results = lib_regression.detection_performance(lr, X_val_for_test, Y_val_for_test, outf)
                if best_auroc < results['TMP']['AUROC']:
                    best_auroc = results['TMP']['AUROC']
                    best_index = score
                    best_result = lib_regression.detection_performance(lr, X_test, Y_test, outf)
            list_best_results_out.append(best_result)
            list_best_results_index_out.append(best_index)
        list_best_results.append(list_best_results_out)
        list_best_results_index.append(list_best_results_index_out)
    '''
    
    dataset_list = [dataset]
    adv_test_list = [adv_type]
    print('evaluate the Mahalanobis estimator')
    score_list = ['Mahalanobis_0.0', 'Mahalanobis_0.01', 'Mahalanobis_0.005', \
                  'Mahalanobis_0.002', 'Mahalanobis_0.0014', 'Mahalanobis_0.001', 'Mahalanobis_0.0005']
    list_best_results_ours, list_best_results_index_ours = [], []
    for dataset in dataset_list:
        print('load train data: ', dataset)
        outf = outf
        list_best_results_out, list_best_results_index_out = [], []
        for out in adv_test_list:
            best_auroc, best_result, best_index = 0, 0, 0
            for score in score_list:
                print('load train data: ', out, ' of ', score)
                total_X, total_Y = lib_regression.load_characteristics(score, dataset, out, outf)
                X_val, Y_val, X_test, Y_test = lib_regression.block_split_adv(total_X, total_Y)
                pivot = int(X_val.shape[0] / 6)
                X_train = np.concatenate((X_val[:pivot], X_val[2*pivot:3*pivot], X_val[4*pivot:5*pivot]))
                Y_train = np.concatenate((Y_val[:pivot], Y_val[2*pivot:3*pivot], Y_val[4*pivot:5*pivot]))
                X_val_for_test = np.concatenate((X_val[pivot:2*pivot], X_val[3*pivot:4*pivot], X_val[5*pivot:]))
                Y_val_for_test = np.concatenate((Y_val[pivot:2*pivot], Y_val[3*pivot:4*pivot], Y_val[5*pivot:]))
                
                lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train)
                y_pred = lr.predict_proba(X_train)[:, 1]
                #print('training mse: {:.4f}'.format(np.mean(y_pred - Y_train)))
                y_pred = lr.predict_proba(X_val_for_test)[:, 1]
                #print('test mse: {:.4f}'.format(np.mean(y_pred - Y_val_for_test)))
                results = lib_regression.detection_performance(lr, X_val_for_test, Y_val_for_test, outf)
                if best_auroc < results['TMP']['AUROC']:
                    best_auroc = results['TMP']['AUROC']
                    best_index = score
                    best_result = lib_regression.detection_performance(lr, X_test, Y_test, outf)
            list_best_results_out.append(best_result)
            list_best_results_index_out.append(best_index)
        list_best_results_ours.append(list_best_results_out)
        list_best_results_index_ours.append(list_best_results_index_out)

    print("evaluation of Mahalanobis estimator (step 1)")

    count_in = 0
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    
    '''
    print("results of LID")
    for in_list in list_best_results:
        print('in_distribution: ' + dataset_list[count_in] + '==========')
        count_out = 0
        for results in in_list:
            print('out_distribution: '+ adv_test_list[count_out])
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            print('\n{val:6.2f}'.format(val=100.*results['TMP']['TNR']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
            print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')
            print('Input noise: ' + list_best_results_index[count_in][count_out])
            print('')
            count_out += 1
        count_in += 1
    '''

    count_in = 0
    print("results of Mahalanobis")
    for in_list in list_best_results_ours:
        print('in_distribution: ' + dataset_list[count_in] + '==========')
        count_out = 0
        for results in in_list:
            print('out_distribution: '+ adv_test_list[count_out])
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
