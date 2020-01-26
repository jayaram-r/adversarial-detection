from __future__ import print_function
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
from constants import ROOT
from helpers.bar import progress_bar


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
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    parser.add_argument('--model-type', default='cifar10', help='model type')
    parser.add_argument('--adv-attack', default='FGSM', help='type of adversarial attack')
    parser.add_argument('--attack', type=bool, default=False, help='launch attack? True or False')
    parser.add_argument('--distance', type=str, default='inf', help='p norm for attack')
    parser.add_argument('--train', type=bool, default=True, help='commence training')
    parser.add_argument('--ckpt', type=bool, default=False, help='use ckpt')
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
             transforms.Normalize((0.1307,), (0.3081,))]
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
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994,0.2010))]
        )
        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994,0.2010))]
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
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
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
        model_path = os.path.join(ROOT, 'models', args.model_type + '_cnn.pt')
        model.load_state_dict(torch.load(model_path))
    
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
        #adversarials = attack_model(images, labels)
    
    if args.save_model:
        model_path = os.path.join(ROOT, 'models', args.model_type + '_cnn.pt')
        torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    main()
