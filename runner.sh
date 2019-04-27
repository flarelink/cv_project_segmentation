#!/bin/bash

python3 main.py --log_name Segnet_Cityscapes_100epochs --n_epochs 100
python3 main.py --model FCN --log_name FCN_Cityscapes_100epochs --n_epochs 100

