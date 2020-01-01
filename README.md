# Semantic Segmentation

Computer vision class final project for semantic segmentation.

Utilized 2 deep neural networks, Fully Convolutional Network (FCN) and SegNet,to perform semantic segmentation on the NYUv2 and Cityscapes datasets.

The metrics utilized to evaluate the performance of the networks were: pixel accuracy, mean pixel accuracy, and intersection over union (IoU).

Some images were shown to successfully segment out portions of the images well. 

For example the image below is from Cityscapes:

<img src="https://github.com/flarelink/cv_project_segmentation/blob/master/images/berlin_000001_000019_leftImg8bit.png" width="512" height="auto" title="Cityscapes Image">

The output using FCN can be seen below:

<img src="https://github.com/flarelink/cv_project_segmentation/blob/master/images/city_fcn_out.png" width="512" height="auto" title="Cityscapes FCN Image">

In comparison, the output using SegNet can be seen below:

<img src="https://github.com/flarelink/cv_project_segmentation/blob/master/images/city_segnet_out.png" width="512" height="auto" title="Cityscapes SegNet Image">
