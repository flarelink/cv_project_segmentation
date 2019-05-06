#!/bin/bash

#python3 main.py --dataset City --log_name Segnet_City_100epochs --n_epochs 100
python3 main.py --dataset City --model FCN --log_name FCN_City_100epochs --n_epochs 100
#python3 main.py --dataset NYUv2 --model FCN --log_name FCN_NYUv2_100epochs --n_epochs 100

