import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os

os.environ["CUDA_VISIBLE_DEVICES"]="2"

transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

net = Net()

import foolbox
#import torchvision.models as models
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Net()
model.load_state_dict(torch.load('../models/mnist_cnn.pt'))
model.to(device)
model.eval()

attacked_model = foolbox.models.PyTorchModel(model, bounds=(0,1), num_classes=10)
FGSM_attack = foolbox.attacks.FGSM(attacked_model)

dataiter = iter(trainloader)
img, label = dataiter.next()
img,label = img.to(device), label.to(device)

img_numpy = img.cpu().numpy()
label_numpy = label.cpu().numpy()

#print (img_numpy.shape, type(img_numpy), label_numpy.shape, type(label_numpy))
print(img.shape, type(img), type(model))
#print (label)
print (model(img))

adversarial = FGSM_attack(img_numpy, label_numpy)
