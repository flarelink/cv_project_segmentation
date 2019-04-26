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
def training(model, run, n_epochs, dataset, trainLoader, testLoader, running_metrics, model_dir):
    """
    Training process for semantic segmentation

    :param model: chosen model, ex) FCNs
    :param run: current run of the program
    :param n_epochs: number of epochs
    :param dataset: chosen dataset, ex) Cityscapes
    :param trainLoader: train dataset loader
    :param testLoader: test dataset loader
    :param running_metrics: class to save metrics over time
    :param model_dir: directory for model output
    :return:
    """

    # set best iou variable to be initialized for testing
    best_iou = 0

    # set loss and optimization method
    criterion = cross_entropy2d
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9,
                          weight_decay=0.0016)  # paper uses stochastic gradient descent

    # run through each epoch
    for epoch in range(n_epochs):

        # grab initial time at each epoch
        begin_epoch = time.time()

        # run through each image relative to batch size
        for iter, data in enumerate(trainLoader, 0):

            # set to train model
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

            # print info after 10 iterations during an epoch
            if iter % 40 == 0:
                print("epoch: {}, iter: {}, loss: {}".format(epoch, iter, loss.item()))

        # print time after each epoch
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - begin_epoch))

        # run through testing on model
        testing(model, run, epoch, dataset, testLoader, running_metrics, best_iou, model_dir)

    # saving model after running through all epochs
    name = os.path.join(model_dir, '{}_{}_final_epoch_at_epoch_{}_on_run_{}.ckpt'.format(model.__class__.__name__, dataset, n_epochs, run))
    torch.save(model.state_dict(), name)


def testing(model, run, epoch, dataset, testLoader, running_metrics, best_iou, model_dir):
    """
    Testing process for semantic segmentation

    :param model: chosen model, ex) FCNs
    :param run: current run of the program
    :param epoch: current epoch
    :param dataset: chosen dataset, ex) Cityscapes
    :param testLoader: testing dataset loader
    :param running_metrics: class to save metrics over time
    :param best_iou: save models with best_iou after 10 epochs
    :param model_dir: directory for model output
    :return:
    """

    # set model to evuation mode
    model.eval()

    # don't want to compute gradients so no grad
    with torch.no_grad():

        # run through each image relative to batch size
        for iter, data in enumerate(testLoader, 0):

            # grab images with labels
            images, labels = data
            images = images.to(device)
            #labels = labels.to(device)

            # run through model
            output = model(images)

            # check predicted vs ground truth
            pred = output.data.max(1)[1].cpu().numpy()
            gt = labels.data.cpu().numpy()

            # update running metrics
            running_metrics.update(gt, pred)

    # get metrics from running metrics
    score, class_iou = running_metrics.get_scores()
    for k, v in score.items():
        print(k, v)
    for k, v in class_iou.items():
        print(k, v)

    # check if best iou was obtained and if epoch > 10, then if true, save that model
    if score["Mean IoU : \t"] > best_iou and epoch >= 10:
        name = os.path.join(model_dir, '{}_{}_best_iou_at_epoch_{}_on_run_{}.ckpt'.format(model.__class__.__name__, dataset, epoch, run))
        torch.save(model.state_dict(), name)

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

    print(model.__class__.__name__ + ' was selected')

    # metrics
    running_metrics = runningScore(n_classes)

    for run in range(args.n_runs):
        # training and validation in one function
        training(model,
                 run,
                 args.n_epochs,
                 args.dataset,
                 trainLoader,
                 testLoader,
                 running_metrics,
                 model_dir)


if __name__ == '__main__':
    main()
