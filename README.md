# EgoNet Model

This is a code repository for the paper "First-Person Action-Object Detection with EgoNet". Our method predicts action-objects from first-person RGB or RGBD data. This work has been published in RSS 2017 Conference.

Citation:  
@InProceedings{gberta_2017_RSS,  
author = {Gedas Bertasius and Hyun Soo Park and Stella X. Yu and Jianbo Shi},  
title = {First-Person Action-Object Detection with EgoNet},  
booktitle = {Proceedings of Robotics: Science and Systems},  
month = {July},  
year = {2017}  
}

## Installation

1. Caffe Deep Learning library and its Python Wrapper (We use DeepLab_v2 version):

	Caffe source code is included. Caffe and its python wrapper need to be compiled as instructed in http://caffe.berkeleyvision.org/installation.html. 


## Usage

Change the path to your caffe directory in the "predict.py" file. Then, change the paths to the locations where the caffe models are placed. Select which of the models you want to run (RGB or RGB + DHG). Finally, run "predict.py"

## Pretrained Models

Please download the pretrained models from the following link: https://drive.google.com/drive/folders/0B5ev-b4SAFZVcWk5emw5UTBtUWs?usp=sharing. We provide two models: 1) a model that works with first-person RGB and DHG inputs, and 2) a model with works just with the first-person RGB images.