# =============================================================================
# segment.py - Performs segmentation on an image given a model, modified from
#              reference
# References:
# - https://github.com/meetshah1995/pytorch-semseg/blob/master/test.py
# =============================================================================

# example runs for segnet and fcn on cityscapes:
# python segment.py --model Segnet --dataset City --model_path ./saved_models/2019-04-29_00_43_16.892291_Segnet_City_final_epoch_at_epoch_100_on_run_0.ckpt --img_path ./cityscapes_dataset/leftImg8bit/test/berlin/berlin_000001_000019_leftImg8bit.png --out_path ./seg_out/city_segnet_out.png
# python segment.py --model FCN --dataset City --model_path ./saved_models/FCNs_City_best_iou_at_epoch_71_on_run_0.ckpt --img_path ./cityscapes_dataset/leftImg8bit/test/berlin/berlin_000001_000019_leftImg8bit.png --out_path ./seg_out/city_segnet_out.png

# example runs for segnet and fcn on nyu
# python segment.py --model Segnet --dataset NYUv2 --model_path ./saved_models/2019-04-29_11_57_46.913750_Segnet_NYUv2_final_epoch_at_epoch_100_on_run_0.ckpt --img_path ./NYUv2/nyu_test_rgb/nyu_rgb_0001.png --out_path ./seg_out/nyu_seg_out.png
# python segment.py --model FCN --dataset NYUv2 --model_path ./saved_models/2019-04-29_19_23_21.294338_FCNs_NYUv2_final_epoch_at_epoch_100_on_run_0.ckpt --img_path ./NYUv2/nyu_test_rgb/nyu_rgb_0001.png --out_path ./seg_out/nyu_fcn_out.png


import os
import torch
import argparse
import numpy as np
import scipy.misc as misc

from utils.cityscapes_loader import *
from utils.nyuv2_loader import *
from models.FCN import *
from models.SegNet import *

def test(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[: model_file_name.find("_")]

    # Setup image
    print("Read Input Image from : {}".format(args.img_path))
    img = misc.imread(args.img_path)

    if(args.dataset == 'City'):
        data_loader = cityscapesLoader
        n_classes = 19
        root_data = './cityscapes_dataset'
    elif(args.dataset == 'NYUv2'):
        data_loader = NYUv2Loader
        n_classes = 14
        root_data = './NYUv2/'
    else:
        raise ValueError('Invalid dataset name. Run python3 segment.py -h to review your options.')

    loader = data_loader(root=root_data, is_transform=True, img_norm=args.img_norm, test_mode=True)

    resized_img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]), interp="bicubic")

    orig_size = img.shape[:-1]
    img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))

    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= loader.mean
    if args.img_norm:
        img = img.astype(float) / 255.0

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    # Setup Model
    if(args.model == 'FCN'):
        vgg_model = VGGNet(requires_grad=True)
        model = FCNs(pretrained_net=vgg_model, n_class=n_classes).to(device)
    elif(args.model == 'Segnet'):
        model = Segnet(n_classes=n_classes, in_channels=3, is_unpooling=True).to(device)
    else:
        raise ValueError('Invalid model name. Run python3 segment.py -h to review your options.')

    state = torch.load(args.model_path)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    images = img.to(device)
    outputs = model(images)


    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    decoded = loader.decode_segmap(pred)
    print("Classes found: ", np.unique(pred))
    misc.imsave(args.out_path, decoded)
    print("Segmentation Mask Saved at: {}".format(args.out_path))


if __name__ == "__main__":

    # create dir for model output
    seg_dir = "seg_out"
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)

    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default=None,
        help="Path to the saved model",
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        default="City",
        help="Dataset to use ['City NYUv2']",
    )

    parser.add_argument(
        "--img_norm",
        dest="img_norm",
        action="store_true",
        help="Enable input image scales normalization [0, 1] \
                              | True by default",
    )
    parser.add_argument(
        "--no-img_norm",
        dest="img_norm",
        action="store_false",
        help="Disable input image scales normalization [0, 1] |\
                              True by default",
    )
    parser.set_defaults(img_norm=True)

    parser.add_argument(
        "--img_path", nargs="?", type=str, default=None, help="Path of the input image"
    )
    parser.add_argument(
        "--out_path", nargs="?", type=str, default=None, help="Path of the output segmap"
    )

    # determine which model to pick
    parser.add_argument('--model', type=str, default='Segnet',
                        help="""Chooses model type:
                                 FCN
                                 Segnet
                                 default=Segnet""")

    args = parser.parse_args()
    test(args)

