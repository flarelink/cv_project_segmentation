#!/bin/bash

python3 main.py --dataset VOC --log_name Segnet_VOC_100epochs --n_epochs 100
python3 main.py --dataset VOC --model FCN --log_name FCN_VOC_100epochs --n_epochs 100

