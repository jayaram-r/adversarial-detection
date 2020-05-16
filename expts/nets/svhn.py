from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class SVHN(nn.Module):
    def __init__(self):
        super(SVHN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.dropout3 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12544, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    def penultimate_forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        pen = self.dropout3(x)
        x = self.fc3(pen)
        output = F.log_softmax(x, dim=1)
        return output, pen


    def intermediate_forward(self, x, layer_index):
        if layer_index == 1:
            x = self.conv1(x)
            x = F.relu(x)
        if layer_index == 2:
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
        if layer_index == 3:
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
        if layer_index == 4:
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.dropout3(x)
        if layer_index == 5:
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.dropout3(x)
            x = self.fc3(x)
        return x

    def layer_wise(self, x):
        # Method to get the layer-wise embeddings for the proposed method
        # Input is included as the first layer
        output = [x]    # 1

        x = self.conv1(x)
        # output.append(x)
        x = F.relu(x)
        output.append(x)    # 2

        x = self.conv2(x)
        # output.append(x)
        x = F.relu(x)
        # output.append(x)
        x = F.max_pool2d(x, 2)
        # output.append(x)
        x = self.dropout1(x)
        output.append(x)    # 3

        x = torch.flatten(x, 1)
        # output.append(x)
        x = self.fc1(x)
        # output.append(x)
        x = F.relu(x)
        # output.append(x)
        x = self.dropout2(x)
        output.append(x)    # 4

        x = self.fc2(x)
        # output.append(x)
        x = F.relu(x)
        # output.append(x)
        x = self.dropout3(x)
        output.append(x)    # 5

        x = self.fc3(x)
        output.append(x)    # 6 (logits)

        # final = F.log_softmax(x, dim=1)
        # output.append(x)
        return output

    def layer_wise_deep_mahalanobis(self, x):
        # Method to get the layer-wise embeddings for the proposed method
        # Input is included as the first layer
        output = [x]    # 1

        x = self.conv1(x)
        # output.append(x)
        x = F.relu(x)
        output.append(x)    # 2

        x = self.conv2(x)
        # output.append(x)
        x = F.relu(x)
        # output.append(x)
        x = F.max_pool2d(x, 2)
        # output.append(x)
        x = self.dropout1(x)
        output.append(x)    # 3

        x = torch.flatten(x, 1)
        # output.append(x)
        x = self.fc1(x)
        # output.append(x)
        x = F.relu(x)
        # output.append(x)
        x = self.dropout2(x)
        output.append(x)    # 4

        x = self.fc2(x)
        # output.append(x)
        x = F.relu(x)
        # output.append(x)
        x = self.dropout3(x)
        output.append(x)    # 5

        x = self.fc3(x)
        output.append(x)    # 6 (logits)

        final = F.log_softmax(x, dim=1)
        # output.append(x)
        return final, output

    def layer_wise_odds_are_odd(self, x):
        # Method to get the latent layer and logit layer outputs for the "odds-are-odd" method
        output = []
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout3(x)
        output.append(x)  # latents
        x = self.fc3(x)
        output.append(x)    # logits
        # final = F.log_softmax(x, dim=1)

        return output

    def layer_wise_lid_method(self, x):
        # Method to get the layer-wise embeddings for the LID adversarial subspaces paper
        # Input is included as the first layer
        output = [x]    # 1
        x = self.conv1(x)
        output.append(x)    # 2
        x = F.relu(x)
        output.append(x)    # 3
        x = self.conv2(x)
        output.append(x)    # 4
        x = F.relu(x)
        output.append(x)    # 5
        x = F.max_pool2d(x, 2)
        output.append(x)    # 6
        x = self.dropout1(x)
        # Skipping this layer and taking its flattened version
        # output.append(x)
        x = torch.flatten(x, 1)
        output.append(x)    # 7
        x = self.fc1(x)
        output.append(x)    # 8
        x = F.relu(x)
        output.append(x)    # 9
        x = self.dropout2(x)
        output.append(x)    # 10
        x = self.fc2(x)
        output.append(x)    # 11
        x = F.relu(x)
        output.append(x)    # 12
        x = self.dropout3(x)
        output.append(x)    # 13
        x = self.fc3(x)
        output.append(x)    # 14 (logits)

        # Skipping this layer because it is simply a shifted version of the logit layer
        # final = F.log_softmax(x, dim=1)
        # output.append(x)
        return output
