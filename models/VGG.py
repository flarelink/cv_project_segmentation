# =============================================================================
# VGG.py - VGG implementation from pytorch library and reference specifically
#          for fully convolutional networks (FCNs) as we get rid of last fully
#          connected layers and extract pooling layers
# References:
# - https://github.com/yunlongdong/FCN-pytorch-easiest
# =============================================================================

from __future__ import print_function, division

import torch.nn as nn
from torchvision.models.vgg import VGG

__all__ = ['VGGNet']


# used for selecting specific VGG architecture
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# used to get features from (X, Y)
# ex) For 'vgg16' the first pooling layer output is at layer 4 so if we loop through the features
#     from 0 to 5 then we get the output after pooling from layer 4
ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

class VGGNet(VGG):

    def __init__(self, pretrained=True, model='vgg16'):
        super().__init__(make_layers(cfg[model]))

        # set specific ranges based off VGG model to get feature extracted pooling layers
        self.ranges = ranges[model]

        # pretrained weights from ImageNet
        if (pretrained):
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        # remove last FC layers so we can replace with deconv layers in FCNs
        del self.classifier

    def forward(self, x):
        output = {}
        # get the output of each max pooling layer (5 max pooling layers in all VGG net)
        # FCN comment below is relative to VGG16
        # for FCN8 we need output after 5th pooling layer
        # for FCN16 we need output after 4th pooling layer
        # for FCN32 we need output after 3rd pooling layer
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d" % (idx + 1)] = x

        return output

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

