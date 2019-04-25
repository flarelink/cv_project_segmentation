# =============================================================================
# main.py - Performs semantic segmentation on provided datasets
# Authors:  Humza Syed & Yuri Yakovlev
# References:
# - https://github.com/yunlongdong/FCN-pytorch-easiest
# - https://github.com/pochih/FCN-pytorch
# =============================================================================

"""
##############################################################################
# Library Imports
##############################################################################
"""
from __future__ import print_function, division

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import time
import os

"""
##############################################################################
# External folder imports
##############################################################################
"""
# models
from models.VGG import *
from models.FCN import *

# utils
from utils.dataloader import *


"""
##############################################################################
# Only set a single global variable for GPU/CPU selection
##############################################################################
"""
# Device configuration to check for cpu or gpu, if gpu detected then use gpu
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(device, " the GPU device is selected")
else:
    device = torch.device('cpu')
    print(device, " the CPU device is selected")

"""
##############################################################################
# Parser
##############################################################################
"""


def create_parser():
    """
    Function to take in input arguments from the user

    return: parsed command line inputs
    """

    # parsing input arguments
    parser = argparse.ArgumentParser(
        description='Semantic segmentation',
        formatter_class=argparse.RawTextHelpFormatter
    )

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            return argparse.ArgumentTypeError('Boolean value expected.')

    # arguments for logging
    parser.add_argument('--log_name', type=str, default='testing',
                        help='Specify log name, if not specified will set to default; default=testing')

    # arguments for determining which dataset to run
    parser.add_argument('--t_dataset', type=int, default=0,
                        help="""Chooses dataset:
                                 0 - Cityscapes 
                                 1 - PASCAL VOC 2011
                                 2 - CamVid  
                                 default=0""")
    parser.add_argument('--n_classes', type=int, default=10,
                        help='Number of classes in dataset; default=10')

    # determine which model to pick
    parser.add_argument('--t_model', type=int, default=0,
                        help="""Chooses model type:
                             0 - FCNs
                             1 - FCN8s 
                             2 - FCN16s 
                             3 - FCN32s   
                             default=0""")

    # arguments for hyperparameters
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Defines num epochs for training; default=100')
    parser.add_argument('--n_runs', type=int, default=1,
                        help='Defines num runs for program; default=1')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Defines batch size for data; default=16')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Defines learning rate for training; default=1e-2')
    parser.add_argument('--decay', type=float, default=0.0004,
                        help='Defines decay for training; default=0.0004')
    parser.add_argument('--random_seed', type=int, default=7,
                        help='Defines random seed value, if set to 0 then randomly sets seed; default=7')

    args = parser.parse_args()

    return args


"""
##############################################################################
# Training, Validation, and Testing
##############################################################################
"""
def training(n_epochs, n_class, model, optimizer, criterion, train_loader, val_loader, IU_scores, pixel_scores, model_dir, score_dir):
    """
    Training process for semantic segmentation

    :param n_epochs: number of epochs
    :param n_class: number of classes
    :param model: chosen model, ex) FCNs
    :param optimizer: input optimizer, ex) SGD
    :param criterion: loss criteria, ex) BCE for semantic segmentation
    :param train_loader: train dataset loader
    :param val_loader: validation dataset loader
    :param IU_scores: Intersection Over Union measurement
    :param pixel_scores: Pixel intersection score
    :param model_dir: directory for model output
    :param score_dir: directory for scores output
    :return:
    """
    # run through each epoch
    for epoch in range(n_epochs):

        # grab initial time at each epoch
        begin_epoch = time.time()

        # run through each image relative to batch size
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()

            # grab images with labels
            inputs, labels = Variable(batch['X']).to(device), Variable(batch['Y']).to(device)

            # run through model, collect loss, then backprop
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print additional information about epoch every 10 epochs
            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.data[0]))

        # print time after each epoch
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - begin_epoch))
        torch.save(model, model_dir)

        val(epoch, n_class, model, val_loader, IU_scores, pixel_scores, score_dir)


def val(epoch, n_class, model, val_loader, IU_scores, pixel_scores, score_dir):
    """
    Training process for semantic segmentation

    :param epoch: number of epochs
    :param n_class: number of classes
    :param model: chosen model, ex) FCNs
    :param val_loader: validation dataset loader
    :param IU_scores: Intersection Over Union measurement
    :param pixel_scores: Pixel intersection score
    :param score_dir: directory for scores output
    :return:
    """
    model.eval()
    total_ious = []
    pixel_accs = []
    for iter, batch in enumerate(val_loader):

        inputs = Variable(batch['X'].cuda())

        output = model(inputs)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

        target = batch['l'].cpu().numpy().reshape(N, h, w)
        for p, t in zip(pred, target):
            total_ious.append(iou(p, t, n_class))
            pixel_accs.append(pixel_acc(p, t))

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
    IU_scores[epoch] = ious
    np.save(os.path.join(score_dir, "meanIU"), IU_scores)
    pixel_scores[epoch] = pixel_accs
    np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)


# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions
def iou(pred, target, n_class):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total


"""
##############################################################################
# Main, where all the magic starts~
##############################################################################
"""
def main():
    """
    Runs through semantic segmentation on datasets
    """

    # load parsed arguments
    args = create_parser()

    if (args.random_seed == 0):
        args.random_seed = random.randint(1, 1000)

    # set reproducible random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # create dir for model
    model_dir = "models_output"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # create dir for score
    score_dir = "scores"
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)

    # determine which model was chosen relative to args.t_model
    model = None
    if(0 <= args.t_model <= 4):
        vgg_model = VGGNet()
        if(args.t_model == 0):
            model = FCNs(pretrained_net=vgg_model, n_class=args.n_classes)
        elif(args.t_model == 2):
            model = FCN8s(pretrained_net=vgg_model, n_class=args.n_classes)
        elif (args.t_model == 3):
            model = FCN16s(pretrained_net=vgg_model, n_class=args.n_classes)
        else: #only other case is 4
            model = FCN32s(pretrained_net=vgg_model, n_class=args.n_classes)

    # set loss and optimization method
    criterion = nn.BCELoss().to(device) # BCE loss since we want pixel by pixel comparison to ground truth
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.7) # paper uses stochastic gradient descent

    # variables to determine scores to measure model performance on validation set
    IU_scores = torch.zeros((args.n_epochs, args.n_class))
    pixel_scores = torch.zeros(args.n_epochs)

    # training and validation
    # TODO - replace the loaders with datasets later
    train_loader = None
    val_loader = None
    training(args.n_epochs,
             args.n_classes,
             model,
             optimizer,
             criterion,
             train_loader,
             val_loader,
             IU_scores,
             pixel_scores,
             model_dir,
             score_dir)


if __name__ == '__main__':
    main()