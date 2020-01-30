import torch
from torchvision import datasets, transforms

def statistics(loader=None):
    mean = 0
    std = 0
    for images, _ in loader:
        print(images.shape)
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return (mean, std)
