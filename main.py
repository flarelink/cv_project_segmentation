# =============================================================================
# main.py - Performs semantic segmentation on provided datasets
# Authors:  Humza Syed & Yuri Yakovlev
# References:
# - https://github.com/yunlongdong/FCN-pytorch-easiest
# - https://github.com/meetshah1995/pytorch-semseg
# =============================================================================

"""
##############################################################################
# Library Imports
##############################################################################
"""
from __future__ import print_function, division

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import time
import os
from datetime import datetime

"""
##############################################################################
# External folder imports
##############################################################################
"""
# models
from models.FCN import *
from models.SegNet import *

# utils
from utils.dataloader import *
from utils.loss import *
from utils.metrics import *
from utils.cityscapes_loader import *
from utils.pascal_voc_loader import *
from utils.nyuv2_loader import *

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
def training(model, run, n_epochs, dataset, trainLoader, testLoader, running_metrics, model_dir, log_file):
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

    # record loss over time
    loss_list = []

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



            # print info after 50 iterations during an epoch
            if iter % 50 == 0:
                print("epoch: {}, iter: {}, loss: {}".format(epoch, iter, loss.item()))
                log_file.write("epoch: {}, iter: {}, loss: {}".format(epoch, iter, loss.item()))

        loss_list.append(loss.item())
        # print time after each epoch
        print("Finished epoch {}, time elapsed {}".format(epoch, time.time() - begin_epoch))
        log_file.write("\nFinished epoch {}, time elapsed {} \n".format(epoch, time.time() - begin_epoch))

        # run through testing on model
        score, class_iou, best_iou = testing(model, run, epoch, dataset, testLoader, running_metrics, best_iou, model_dir, log_file)

    # saving model after running through all epochs
    today_time = str(datetime.today()).replace(':', '_').replace(' ', '_')
    name = '{}_{}_{}_final_epoch_at_epoch_{}_on_run_{}.ckpt'.format(today_time, model.__class__.__name__, dataset, n_epochs, run)
    path = os.path.join(model_dir, name)
    torch.save(model.state_dict(), path)

    return score, class_iou, loss_list, name


def testing(model, run, epoch, dataset, testLoader, running_metrics, best_iou, model_dir, log_file):
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
        log_file.write('{} {} \n'.format(k, v))

    log_file.write('\n class iou \n')
    for k, v in class_iou.items():
        print(k, v)
        log_file.write('{} {} \n'.format(k, v))
    running_metrics.reset()

    # check if best iou was obtained and if epoch > 10, then if true, save that model
    if score["Mean IoU : \t"] > best_iou and epoch >= 49:
        name = os.path.join(model_dir, '{}_{}_best_iou_at_epoch_{}_on_run_{}.ckpt'.format(model.__class__.__name__, dataset, epoch, run))
        torch.save(model.state_dict(), name)

    return score, class_iou, best_iou


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
    parser.add_argument('--dataset', type=str, default='NYUv2',
                        help="""Chooses dataset:
                                 City 
                                 NYUv2
                                 default=NYUv2""")

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
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Defines num epochs for training; default=100')
    parser.add_argument('--n_runs', type=int, default=1,
                        help='Defines num runs for program; default=1')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Defines batch size for data; default=1')
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
    model_dir = "saved_models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # create dir for loss plots
    loss_dir = "loss_plots"
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    # create dir for loss plots
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # open log files and fill in with data
    today_time = str(datetime.today()).replace(':', '_').replace(' ', '_')
    csv_file = open('./logs/log_run_{}_{}.csv'.format(today_time, args.log_name), 'w+')
    csv_file.write('dataset, model, num_epochs, num_of_runs, current_run, batch_size, '
                   'Overall Acc, Mean Acc, FreqW Acc, Mean IoU \n')
    log_file = open('./logs/log_run_{}_{}.txt'.format(today_time, args.log_name), 'w+')
    log_file.write('dataset     = {} \n'.format(args.dataset))
    log_file.write('model       = {} \n'.format(args.model))
    log_file.write('n_epochs    = {} \n'.format(args.n_epochs))
    log_file.write('n_runs      = {} \n'.format(args.n_runs))
    log_file.write('batch_size  = {} \n'.format(args.batch_size))


    ################ PICK DATASET ################
    # determine dataset being used
    if(args.dataset == 'City'):
        #trainLoader, _, testLoader = load_data_Cityscapes(args.batch_size)

        t_loader = cityscapesLoader('./cityscapes_dataset',
                                        is_transform=True, 
                                        split='train',
                                        #img_size = (256, 256)
                                        )

        v_loader = cityscapesLoader(
        root='./cityscapes_dataset',
        is_transform=True,
        split='val',
        #img_size=(256, 256)
        )

        trainLoader = torch.utils.data.DataLoader(
        t_loader,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        )

        testLoader = torch.utils.data.DataLoader(
            v_loader, batch_size=args.batch_size, num_workers=4
        )

        n_classes = 19
    elif(args.dataset == 'NYUv2'):
        t_loader = NYUv2Loader(root='./NYUv2/',
                                   is_transform=True,
                                   split='training'
                                    )

        v_loader = NYUv2Loader(
            root='./NYUv2/',
            is_transform=True,
            split='val',
        )

        trainLoader = torch.utils.data.DataLoader(
            t_loader,
            batch_size=args.batch_size,
            num_workers=4,
            shuffle=True
        )

        testLoader = torch.utils.data.DataLoader(
            v_loader, batch_size=args.batch_size, num_workers=4
        )
        n_classes = 14
    else:
        raise ValueError('Invalid dataset name. Run python3 main.py -h to review your options.')

    ################ PICK MODEL ################
    # determine which model was chosen
    vgg_model = VGGNet(requires_grad=True)
    if (args.model == 'FCN'):
        model = FCNs(pretrained_net=vgg_model, n_class=n_classes).to(device)
    elif (args.model == 'FCN8'):
        model = FCN8s(pretrained_net=vgg_model, n_class=n_classes).to(device)
    elif (args.model == 'FCN16'):
        model = FCN16s(pretrained_net=vgg_model, n_class=n_classes).to(device)
    elif (args.model == 'FCN32'):
        model = FCN32s(pretrained_net=vgg_model, n_class=n_classes).to(device)
    elif (args.model == 'Segnet'):
        model = Segnet(n_classes=n_classes, in_channels=3, is_unpooling=True).to(device)
    else:
        raise ValueError('Invalid model name. Run python3 main.py -h to review your options.')

    print(model.__class__.__name__ + ' was selected')

    # metrics
    running_metrics = runningScore(n_classes)

    ################ TRAINING ################
    for run in range(args.n_runs):
        log_file.write('Current run is: {} \n'.format(run))

        # training and validation in one function
        score, class_iou, loss_list, name = training(model,
                                                     run,
                                                     args.n_epochs,
                                                     args.dataset,
                                                     trainLoader,
                                                     testLoader,
                                                     running_metrics,
                                                     model_dir,
                                                     log_file)
        # record metrics for logging
        metric_vals = []
        for metric, val in score.items():
            metric_vals.append(val)

        csv_file.write('{}, {}, {}, {}, {}, {}, '
                       '{}, {}, {}, {} \n'.format(args.dataset,
                                                   args.model,
                                                   args.n_epochs,
                                                   args.n_runs,
                                                   run,
                                                   args.batch_size,
                                                   metric_vals[0],
                                                   metric_vals[1],
                                                   metric_vals[2],
                                                   metric_vals[3]
                                                   ))

        log_file.write('\n Final metrics: \n'.format(metric_vals[0]))
        log_file.write('Overall Acc = {} \n'.format(metric_vals[0]))
        log_file.write('Mean Acc    = {} \n'.format(metric_vals[1]))
        log_file.write('FreqW Acc   = {} \n'.format(metric_vals[2]))
        log_file.write('Mean IoU    = {} \n'.format(metric_vals[3]))

        name = '{}_{}'.format(today_time, args.log_name)
        # plot loss over time
        loss_plotter(loss_list, name)

if __name__ == '__main__':
    main()
