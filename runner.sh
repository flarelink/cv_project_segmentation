#!/bin/bash

python3 main.py --log_name Segnet_Cityscapes_200epochs --n_epochs 200
python3 main.py --model FCN --log_name FCN_Cityscapes_200epochs --n_epochs 200

