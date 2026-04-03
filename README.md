# GNR-638 Assignment 3: DeepLabv3+ Implementation From Scratch

**Author:** Rakshana Sundaram  
**Roll Number:** 24B2428  

## Overview
This repository contains a "from scratch" PyTorch implementation of the DeepLabv3+ semantic segmentation architecture, as proposed in the paper *"Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"* (ECCV 2018). 

The objective of this assignment is to structurally recreate the core components of the network—specifically Depthwise Separable Convolutions, the Atrous Spatial Pyramid Pooling (ASPP) module, and the Encoder-Decoder fusion block—without relying on pre-built segmentation libraries.

## Repository Contents
* `GNR638_ASSIGNMENT_3_FINAL.ipynb`: The main Jupyter/Colab notebook containing the custom PyTorch implementation, the training loop, and the evaluation code.
* `GNR638_ASSIGNMENT_3_REPORT.pdf`: The final report detailing the methodology, architecture, and a visual evaluation of the results.
* `custom_deeplabv3plus_weights_final.pth`: The serialized PyTorch state dictionary containing the learned weights from the overfitting exercise.

## Dataset and Execution Notes
To accommodate compute and version control constraints, this implementation is designed to be validated on a small "toy" sample of the **PASCAL VOC 2012** dataset. 

**You do not need to download the dataset manually.**
The PyTorch `DataLoader` in the notebook is configured with `download=True`. When you execute the data loading cell, it will automatically download and extract the necessary PASCAL VOC files into a local `./data` directory within your runtime environment.

### How to Run:
1. Clone this repository to your local machine or open the `.ipynb` file in Google Colab.
2. Run the cells sequentially. The script will automatically fetch the data, isolate a sample batch, train the custom architecture to convergence (100 epochs), and generate a visual evaluation of the segmentation masks.

## Results Summary
The custom implementation successfully learned to map spatial features and overfit the training sample, proving the structural validity of the custom ASPP and decoder modules. The included report provides a visual evaluation comparing the model outputs to verify that the core mechanics of the architecture effectively capture spatial features.
