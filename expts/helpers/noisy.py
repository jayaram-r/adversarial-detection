import torchvision
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR

def get_noisy(model, device, test_loader, criterion=None):
    model.eval()
    
    len_test_loader = len(test_loader)
    len_data = len(test_loader.dataset)
    test_loss = 0.
    correct = 0.
    total = 0.
    
    accuracy = []
    
    threshold1 = 0.1/100 #0.1% threshold
    threshold2 = 0.5/100 #0.5% threshold

    std_dev_list = [float(i)/255 for i in range(1,21,2)]
    
    with torch.no_grad():
        for std_dev in std_dev_list:
            for batch_idx, (data, target) in enumerate(test_loader):
                # print(data.shape, target.shape, type(data), type(target))
                shape = tuple(list(data.shape))
                rand = torch.normal(mean=0., std=std_dev, size=shape)
                data = data + rand
                data, target = data.to(device), target.to(device)
                output = model(data)
                if criterion is None:
                    test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    test_loss /= len_data
                else:
                    loss = criterion(output, target)
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                    test_loss += loss.item()
            
            cur_accuracy = float(correct)/len_data * 100
            accuracy.append(cur_accuracy)
            if len(accuracy) > 1:
                if accuracy[0] - cur_accuracy >= threshold1:
                    return std_dev
    
    #process list to return best std_dev value
    for i in range(1, len(accuracy)):
        if abs(accuracy[0] - accuracy[i]) >= threshold2:
            return std_dev_list[i]
    
    return None


def create_noise_samples(loader, std_dev):
    X = []
    Y = []
    for batch_dix, (data, target) in enumerate(loader):
        shape = tuple(list(data.shape))
        rand = torch.normal(mean=0., std=std_dev, size=shape)
        data = data + rand
        data, target = data.cpu().numpy(), target.cpu().numpy()
        if batch_idx == 0:
            X, Y = data, target
        else:
            X = np.vstack((X, data))
            Y = np.vstack((Y, target))

    Y = Y.reshape((-1,))
    return X, Y
