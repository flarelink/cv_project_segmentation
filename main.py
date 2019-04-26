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
from datetime import datetime

"""
##############################################################################
# External folder imports
##############################################################################
"""
# models
from models.VGG import *
from models.FCN import *
from models.SegNet import *

# utils
from utils.dataloader import *
from utils.loss import *
from utils.metrics import *

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

    running_loss = 0

    # run through each epoch
    for epoch in range(n_epochs):

        # grab initial time at each epoch
        begin_epoch = time.time()

        # run through each image relative to batch size
        for iter, data in enumerate(train_loader, 0):

            model.train()

            # grab images with labels
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # set gradient to zero
            optimizer.zero_grad()

            # run through model, collect loss, then backprop
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print info after around 10 iterations during an epoch
            if iter % 10 == 0:
                print("epoch: {}, iter: {}, loss: {}".format(epoch, iter, loss.item()))

        # print time after each epoch
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - begin_epoch))
        if(epoch != 0 and epoch % 50 == 0):
            # saving model after running through all epochs
            torch.save(model, model_dir)

        val(epoch, n_class, model, val_loader, IU_scores, pixel_scores, score_dir)

    # saving model after running through all epochs
    today_time = str(datetime.today()).replace(':', '_').replace(' ', '_')
    name = os.path.join(model_dir, '_' + today_time + '_model.ckpt')
    torch.save(model.state_dict(), name)



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

    for iter, data in enumerate(val_loader, 0):

        # grab images with labels
        images, labels = data
        images = images.to(device)
        #labels = labels.to(device)

        output = model(images)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

        target = labels.cpu().numpy().reshape(N, h, w)
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
    parser.add_argument('--dataset', type=str, default='City',
                        help="""Chooses dataset:
                                 City 
                                 VOC
                                 default=City""")

    # determine which model to pick
    parser.add_argument('--model', type=str, default='Segnet',
                        help="""Chooses model type:
                             FCN
                             FCN8
                             FCN16
                             FCN32 
                             Segnet
                             default=Segnet""")

    # arguments for hyperparameters
    parser.add_argument('--n_epochs', type=int, default=3,
                        help='Defines num epochs for training; default=3')
    parser.add_argument('--n_runs', type=int, default=1,
                        help='Defines num runs for program; default=1')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Defines batch size for data; default=10')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Defines learning rate for training; default=1e-3')
    parser.add_argument('--random_seed', type=int, default=7,
                        help='Defines random seed value, if set to 0 then randomly sets seed; default=7')

    args = parser.parse_args()

    return args


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

    # set reproducible random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    # create dir for model output
    model_dir = "model_output"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # create dir for score
    score_dir = "scores"
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)

    # determine dataset being used
    if(args.dataset == 'City'):
        trainLoader, _, testLoader = load_data_Cityscapes(args.batch_size)
        n_classes = 19
    elif(args.dataset == 'VOC'):
        trainLoader, testLoader = load_data_VOC(args.batch_size)
        n_classes = 21
    else:
        raise ValueError('Invalid dataset name. Run python3 main.py -h to review your options.')

    # determine which model was chosen
    #vgg_model = VGGNet()
    if (args.model == 'FCN'):
        model = FCNs(pretrained_net=vgg_model, n_class=n_classes)
    elif (args.model == 'FCN8'):
        model = FCN8s(pretrained_net=vgg_model, n_class=n_classes)
    elif (args.model == 'FCN16'):
        model = FCN16s(pretrained_net=vgg_model, n_class=n_classes)
    elif (args.model == 'FCN32'):
        model = FCN16s(pretrained_net=vgg_model, n_class=n_classes)
    elif (args.model == 'FCN32'):
        model = FCN16s(pretrained_net=vgg_model, n_class=n_classes)
    elif (args.model == 'Segnet'):
        model = Segnet(n_classes=n_classes, in_channels=3, is_unpooling=True).to(device)
    else:
        raise ValueError('Invalid model name. Run python3 main.py -h to review your options.')

    # set loss and optimization method
    criterion = cross_entropy2d
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0016) # paper uses stochastic gradient descent

    # metrics
    metrics = runningScore(n_classes)

    # variables to determine scores to measure model performance on validation set
    IU_scores = np.zeros((args.n_epochs, n_classes))
    pixel_scores = np.zeros(args.n_epochs)

    # training and validation
    # TODO - replace the loaders with datasets later
    training(args.n_epochs,
             n_classes,
             model,
             optimizer,
             criterion,
             trainLoader,
             testLoader,
             IU_scores,
             pixel_scores,
             model_dir,
             score_dir)


if __name__ == '__main__':
    main()
