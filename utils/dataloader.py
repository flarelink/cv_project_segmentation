from __future__ import print_function, division

import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
plt.ion()

# make sure user imports only the functions
__all__ = ['load_data_Cityscapes', 'load_data_VOC']


#TODO:  Look at these loaders and figure out how to morph them to the code
#       They are very new, came out sometime early 2019 and seem to have no
#       usages online.
def load_data_Cityscapes(batch_size):
    """
    Load Cityscapes dataloaders
    :param batch_size: batch size for datasets
    :return: train and test dataloaders for Cityscapes
    """

    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
    ])

    trainset = torchvision.datasets.Cityscapes(root='./cityscapes_dataset',
                                               split='train',
                                               mode='fine',
                                               target_type='semantic',
                                               transform=transform,
                                               target_transform=transform)
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)

    valset = torchvision.datasets.Cityscapes(root='./cityscapes_dataset',
                                               split='val',
                                               mode='fine',
                                               target_type='semantic',
                                               transform=transform,
                                               target_transform=transform)
    valLoader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.Cityscapes(root='./cityscapes_dataset',
                                              split='test',
                                              mode='fine',
                                              target_type='semantic',
                                              transform=transform,
                                              target_transform=transform)

    testLoader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=4)

    return trainLoader, valLoader, testLoader


def load_data_VOC(batch_size):
    """
    Load VOC dataloaders
    :param batch_size: batch size for datasets
    :return: train and test dataloaders for VOC
    """

    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
    ])

    trainset = torchvision.datasets.VOCSegmentation(root='./VOC',
                                                    image_set='trainval',
                                                    download=True,
                                                    transform=transform,
                                                    target_transform=transform)

    #trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #                                          shuffle=True, num_workers=4)

    testset = torchvision.datasets.VOCSegmentation(root='./VOC',
                                                   image_set='val',
                                                   download=True,
                                                   transform=transform,
                                                   target_transform=transform)

    #testLoader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                         shuffle=False, num_workers=4)

    #return trainLoader, testLoader
