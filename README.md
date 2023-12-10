# MulticlassSegment

This project focuses on performing multiclass segmentation on images using two different neural network architectures: U-Net and a custom-designed segmentation network. The goal is to accurately segment objects in images into various classes.

## Overview

Program language: Python

Framework: tensorflow

Model architecture:

In this project, I explore two different neural network architectures for multiclass segmentation:

1. U-Net: consists of an encoder-decoder structure that captures both local and global features for accurate segmentation.

2. Custom Segmentation Network: Apart from U-Net, I design and implement a custom neural network architecture tailored to the specific multiclass segmentation task.

## Installation

pip install -r requirements.txt

## How to use

Training 

python /path/to/train.py --config "/path/to/config/file"

Test

python /path/to/test.py --config "/path/to/config/file"

## Result

![](results/0000012.png)
![](results/0000044.png)
![](results/0000105.png)
![](results/0000145.png)
![](results/0000328.png)
![](results/0000089.png)


