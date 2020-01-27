from __future__ import absolute_import, division, print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from nets.mnist import *
from nets.cifar10 import *
from nets.svhn import *
from nets.resnet import *
import os
import foolbox
from .constants import ROOT, NORMALIZE_IMAGES
from helpers.bar import progress_bar
from .utils import (
    load_model_checkpoint,
    save_model_checkpoint
)


def train(args, model, device, train_loader, optimizer, epoch, criterion=None):
    model.train()

    len_train_loader = len(train_loader)
    len_data = train_loader.dataset.targets.size(0)
    train_loss = 0.
    correct = 0.
    total = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if criterion is not None:
            loss = criterion(output, target)
        else:
            loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and criterion is None:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.
                  format(epoch, batch_idx * target.size(0), len_data,
                         100. * batch_idx / len_train_loader, loss.item()))

        if criterion is not None:
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        if batch_idx % args.log_interval == 0 and criterion is not None:
            progress_bar(batch_idx, len_train_loader, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return model


def test(args, model, device, test_loader, criterion=None):
    model.eval()

    len_test_loader = len(test_loader)
    len_data = test_loader.dataset.targets.size(0)
    test_loss = 0.
    correct = 0.
    total = 0.
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # print(data.shape, target.shape, type(data), type(target))
            data, target = data.to(device), target.to(device)
            output = model(data)
            if criterion is None:
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                # TODO: Check if this correct
                test_loss /= len_data
            else:
                loss = criterion(output, target)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                test_loss += loss.item()
                # test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
                progress_bar(batch_idx, len_test_loader, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    if criterion is None:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
              format(test_loss, correct, len_data, 100. * correct / len_data))
    # else:
    #    print('Accuracy of the network on the %d test images: %d %%' % (total, 100. * correct / total))

    return model


def attack(attack_model, device, test_loader):
    # model.eval()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data_numpy = data.data.cpu().numpy()
        target_numpy = target.data.cpu().numpy()
        adversarials = attack_model(data_numpy, target_numpy) #, target_numpy, unpack=False)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--model-type', '-m', choices=['mnist', 'cifar10', 'svhn'], default='cifar10',
                        help='model type or name of the dataset')
    parser.add_argument('--batch-size', '-b', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', '--tb', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', '-e', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', '-g', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', '-s', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='number of batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    parser.add_argument('--adv-attack', '--aa', choices=['FGSM', 'PGD', 'CW'], default='FGSM',
                        help='type of adversarial attack')
    parser.add_argument('--attack', action='store_true', default=False, help='option to launch adversarial attack')
    parser.add_argument('--distance', '-d', type=str, default='inf', help='p norm for attack')
    parser.add_argument('--train', '-t', action='store_true', default=True, help='commence training')
    parser.add_argument('--ckpt', action='store_true', default=False, help='Use the saved model checkpoint')
    parser.add_argument('--gpu', type=str, default='2', help='gpus to execute code on')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    data_path = os.path.join(ROOT, 'data')
    criterion = None
    if args.model_type == 'mnist':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(*NORMALIZE_IMAGES['mnist'])]
        )
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train=True, download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True, **kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train=False, transform=transform),
            batch_size=args.test_batch_size, shuffle=True, **kwargs
        )
        model = MNIST().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        bounds = (-255, 255)
        num_classes = 10

    elif args.model_type == 'cifar10':
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(*NORMALIZE_IMAGES['cifar10'])]
        )
        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(*NORMALIZE_IMAGES['cifar10'])]
        )
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        model = ResNet18().to(device)
        criterion = nn.CrossEntropyLoss()
        # Settings recommended in: https://github.com/kuangliu/pytorch-cifar
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer, [150, 250, 350], gamma=0.1)
        bounds = (-255, 255)
        num_classes = 10

    elif args.model_type == 'svhn':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(*NORMALIZE_IMAGES['svhn'])]
        )
        trainset = torchvision.datasets.SVHN(root=data_path, split='train', download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        testset = torchvision.datasets.SVHN(root=data_path, split='test', download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        model = SVHN().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        bounds = (-255, 255)
        num_classes = 10

    else:
        print(args.model_type + " not in candidate models to be trained; halt!")
        exit()
   
    if args.train:
        for epoch in range(1, args.epochs + 1):
            model = train(args, model, device, train_loader, optimizer, epoch, criterion=criterion)
            model = test(args, model, device, test_loader, criterion=criterion)
            scheduler.step()
   
    elif args.ckpt:
        model = load_model_checkpoint(model, args.model_type)
    
    if args.attack:
        distance = None
        if args.distance == '2':
            distance = foolbox.distances.Linf
        elif args.distance == 'inf':
            distance = foolbox.distances.Linf
        else:
            print(args.distance + " not in candidate distances; halt!")
            exit()
    
        model.to(device)
        model.eval()
        fmodel = foolbox.models.PyTorchModel(model, bounds=bounds, num_classes=num_classes)
        
        if args.adv_attack == 'FGSM':
            attack_model = foolbox.attacks.FGSM(fmodel, distance=distance) #distance=foolbox.distances.Linf)
        if args.adv_attack == 'PGD':
            attack_model = foolbox.attacks.RandomStartProjectedGradientDescentAttack(fmodel, distance=distance)
        if args.adv_attack == 'CW':
            attack_model = foolbox.attacks.CarliniWagnerL2Attack(fmodel, distance=distance)
        else:
            print(args.adv_attack + " not in candidate attacks; halt!")
            exit()

        attack(attack_model, device, test_loader)
        # adversarials = attack_model(images, labels)
    
    if args.save_model:
        save_model_checkpoint(model, args.model_type)


if __name__ == '__main__':
    main()
