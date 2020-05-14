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

import torch
import numpy as np
import os

from torch.autograd import Variable

def get_mahalanobis_scores(model, adv_type, dataset, num_labels, outf,
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
                    num_classes, outf, net_type, sample_mean, precision, i, magnitude)
            M_in = np.asarray(M_in, dtype=np.float32)
            if i == 0:
                Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
            else:
                Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)

        for i in range(num_output):
            M_out = lib_generation.get_Mahalanobis_score_adv(model, test_adv_data, test_label, 
                    num_classes, outf, net_type, sample_mean, precision, i, magnitude)
            M_out = np.asarray(M_out, dtype=np.float32)
            if i == 0:
                Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
            else:
                Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)
        
        for i in range(num_output):
            M_noisy = lib_generation.get_Mahalanobis_score_adv(model, test_noisy_data, test_label, 
                    num_classes, outf, net_type, sample_mean, precision, i, magnitude)
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
        
        return Mahalanobis_out

