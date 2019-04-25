from __future__ import print_function, division

import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
plt.ion()

# make sure user imports only the functions
__all__ = ['load_data_Cityscapes', 'load_data_VOC']


def load_data_Cityscapes(batch_size):
    """
    Load Cityscapes dataloaders
    :param batch_size: batch size for datasets
    :return: train and test dataloaders for Cityscapes
    """

    transform = transforms.Compose([
    transforms.ToTensor()
    ])

    trainset = torchvision.datasets.Cityscapes(root='./gtFine',
                                               split='train',
                                               mode='gtFine',
                                               target_type='semantic',
                                               transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.Cityscapes(root='./gtFine',
                                              split='test',
                                              mode='gtFine',
                                              target_type='semantic',
                                              transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=4)

    return trainloader, testloader


def load_data_VOC(batch_size):
    """
    Load VOC dataloaders
    :param batch_size: batch size for datasets
    :return: train and test dataloaders for VOC
    """

    transform = transforms.Compose([
    transforms.ToTensor()
    ])

    trainset = torchvision.datasets.VOCSegmentation(root='./VOC',
                                                    image_set='train',
                                                    download=True,
                                                    transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.VOCSegmentation(root='./VOC',
                                                   image_set='test',
                                                   download=True,
                                                   transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=4)

    return trainloader, testloader