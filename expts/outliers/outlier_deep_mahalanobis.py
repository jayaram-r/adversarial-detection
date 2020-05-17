from __future__ import print_function
from nets.svhn import *
from nets.cifar10 import *
from nets.mnist import *
from nets.resnet import *
import argparse
import torch
import data_loader
import numpy as np
import detectors.calculate_log as callog
import models
import os
import detectors.lib_generation as lib_generation
import detectors.lib_regression as lib_regression

from torchvision import transforms
from torch.autograd import Variable

from sklearn.linear_model import LogisticRegressionCV

def get_outlier_meta_files(model, device, train_loader, test_loader, out_test_loader, out, num_classes, dataset, net_type, outf):
    '''
    train_loader: dataloader with training samples (in-distribution)
    test_loader: dataloader with testing samples (in-distribution)
    num_classes: number of classes for the in-distribution dataset
    out_test_loader: dataloader with testing samples (out-of-distribution)
    note: input dimensionality should be the same for in-distribution and out-of-distribution samples
    '''
    
    out_dist_list = [out]
    
    # set information about feature extaction
    model.eval()
    #lines below need to be varied depending on the dataset for which outlier analysis is to be done
    temp_x = torch.rand(2,3,32,32).cuda()
    temp_x = Variable(temp_x)
    temp_list = model.layer_wise_deep_mahalanobis(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
        
    print('get sample mean and covariance')
    sample_mean, precision = lib_generation.sample_estimator(model, num_classes, feature_list, train_loader)
    
    print('get Mahalanobis scores')
    m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    for magnitude in m_list:
        print('Noise: ' + str(magnitude))
        for i in range(num_output):
            M_in = lib_generation.get_Mahalanobis_score(model, test_loader, num_classes, outf, \
                                                        True, net_type, sample_mean, precision, i, magnitude)
            M_in = np.asarray(M_in, dtype=np.float32)
            if i == 0:
                Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
            else:
                Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)
            
        for out_dist in out_dist_list:
            #out_test_loader = data_loader.getNonTargetDataSet(out_dist, batch_size, in_transform, dataroot)
            print('Out-distribution: ' + out_dist) 
            for i in range(num_output):
                M_out = lib_generation.get_Mahalanobis_score(model, out_test_loader, num_classes, outf, \
                                                             False, net_type, sample_mean, precision, i, magnitude)
                M_out = np.asarray(M_out, dtype=np.float32)
                if i == 0:
                    Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
                else:
                    Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)

            Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
            Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
            Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_in)
            file_name = os.path.join(outf, 'Mahalanobis_%s_%s_%s.npy' % (str(magnitude), dataset , out_dist))
            Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
            np.save(file_name, Mahalanobis_data)

def get_outlier_statistics(in_dataset, out, outf):
    # initial setup
    dataset_list = [in_dataset]
    score_list = ['Mahalanobis_0.0', 'Mahalanobis_0.01', 'Mahalanobis_0.005', 'Mahalanobis_0.002', 'Mahalanobis_0.0014', 'Mahalanobis_0.001', 'Mahalanobis_0.0005']

    # train and measure the performance of Mahalanobis detector
    list_best_results, list_best_results_index = [], []
    for dataset in dataset_list:
        print('In-distribution: ', dataset)
        #outf = './output/' + args.net_type + '_' + dataset + '/'
        outf = outf

        out_list = [out]

        list_best_results_out, list_best_results_index_out = [], []
        for out in out_list:
            print('Out-of-distribution: ', out)
            best_tnr, best_result, best_index = 0, 0, 0
            for score in score_list:
                total_X, total_Y = lib_regression.load_characteristics(score, dataset, out, outf) #need to modify this file to return outlier
                X_val, Y_val, X_test, Y_test = lib_regression.block_split(total_X, total_Y, out)
                X_train = np.concatenate((X_val[:500], X_val[1000:1500]))
                Y_train = np.concatenate((Y_val[:500], Y_val[1000:1500]))
                X_val_for_test = np.concatenate((X_val[500:1000], X_val[1500:]))
                Y_val_for_test = np.concatenate((Y_val[500:1000], Y_val[1500:]))
                lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train)
                y_pred = lr.predict_proba(X_train)[:, 1]
                #print('training mse: {:.4f}'.format(np.mean(y_pred - Y_train)))
                y_pred = lr.predict_proba(X_val_for_test)[:, 1]
                #print('test mse: {:.4f}'.format(np.mean(y_pred - Y_val_for_test)))
                results = lib_regression.detection_performance(lr, X_val_for_test, Y_val_for_test, outf)
                if best_tnr < results['TMP']['TNR']:
                    best_tnr = results['TMP']['TNR']
                    best_index = score
                    best_result = lib_regression.detection_performance(lr, X_test, Y_test, outf)
            list_best_results_out.append(best_result)
            list_best_results_index_out.append(best_index)
        list_best_results.append(list_best_results_out)
        list_best_results_index.append(list_best_results_index_out)

    # print the results
    count_in = 0
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']

    for in_list in list_best_results:
        print('in_distribution: ' + dataset_list[count_in] + '==========')
        out_list = ['svhn', 'imagenet_resize', 'lsun_resize']
        if dataset_list[count_in] == 'svhn':
            out_list = ['cifar10', 'imagenet_resize', 'lsun_resize']
        count_out = 0
        for results in in_list:
            print('out_distribution: '+ out_list[count_out])
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

