# Retinal Vessel Segmentation Using U-Net

## Introduction

This repository contains the code and results for a project on retinal vessel segmentation using U-Net architecture. The goal of this project is to automatically segment and analyze the blood vessels in retinal images, which can be useful for the diagnosis and treatment of various eye and cardiovascular diseases.

## Problem Definition

The DRIVE database was created to facilitate comparative research on the segmentation of blood vessels in retinal pictures. Retinal vessel segmentation and delineation of morphological characteristics of retinal blood vessels, such as length, width, tortuosity, branching patterns, and angles, are utilized for the diagnosis, screening, treatment, and evaluation of various cardiovascular and ophthalmologic diseases, including diabetes, hypertension, arteriosclerosis, and choroidal neovascularization. Automatic identification and analysis of the vasculature may help in the establishment of screening programs for diabetic retinopathy, vascular tortuosity and hypertensive retinopathy, vessel diameter measurement in connection to the diagnosis of hypertension, and computer-assisted laser surgery. For temporal or multimodal image registration and retinal image mosaic synthesis, automatic development of retinal maps and extraction of branch points are used. In addition, the retinal vascular tree is unique to every person and may be utilized for biometric identification.

## Methods

To tackle this problem, we used U-Net architecture to segment the vessels. U-Net is known as one of the leading models used for biomedical image segmentation. We used the following paper as our guide for this project. Furthermore, following implementations of the U-Net on GitHub  helped us to gain insight. 

In order to use the data more efficiently, we generated dataset_loader.py. This piece of code first reads the whole data from the DRIVE folder and stores it in some files. npy format. Later we use these files to load our data to the code. As a part of preprocessing, we also normalize our data before saving it by dividing it by 255 for images and 1 for labels.

In model_scripts, we have the following classes to make our U-Net architecture. In these classes, we used the concepts we learned, such as pooling, relu, batch normalization, etc.

- ConvBNReLU: To implement the structure, we have a flag in this class. The structure would be as follows if isBN is valid: Following ReLu activation, we calculate a 2-D convolution ConvNoPool. Otherwise, the structure will be as follows: ConvNoPool 2-D Batch normalization and ReLu activation are used to construct a 2-D convolution.
- ConvNoPool: ConvBNReLU was implemented twice.
- ConvPool: We started with a 2D Max pool with a kernel size of 2 and a stride of 2, then added two ConvBNReLUs.
- UpsampleConv: In this class, we have two flags depending on the condition, and we will have two structures. If is_deconv is set to true, then: convTranspoe2d will be returned. Otherwise: We will start using UpsamplingBilinear2d and then go on to conv2d. We have two ConvBNReLUs for conv twice.
- ConvOut: We have three Conv2d in this class, with a Sigmoid at the end for the outcome. 
- Unet:  The immediate implementation of structure with a high degree of abstraction may be found by calling preceding classes. 

We evaluate our model using precision, recall, and f1 score, as well as batch area under curve, in print metrics evaluation. We take care of saving the result in a particular iteration using the paste and save method.

We used different optimization algorithms for this project, including Adam and SGDs. You can find the results for ten batches for each of these optimizations in the following. Also, we provided metrics such as precision, recall, f1_scoree, and area under the curve.

Overall, SGD seems to have the most promising performance.

## References

- W. Weng and X. Zhu, "U-Net: Convolutional Networks for Biomedical Image Segmentation," *IEEE Access*, vol. 9, pp. 16591–16603, May 2015, doi: [10.48550/arxiv.1505.04597].
- W. Weng and X. Zhu, "INet: Convolutional Networks for Biomedical Image Segmentation," *IEEE Access*, vol. 9, pp. 16591–16603, 2021, doi: [10.1109/ACCESS.2021.3053408].
- [DRIVE: Digital Retinal Images for Vessel Extraction](https://www.isi.uu.nl/Research/Databases/DRIVE/)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Retinal Vessel Segmentation Using U-Net Architecture](https://www.researchgate.net/publication/344704945_Retinal_Vessel_Segmentation_Using_U-Net_Architecture)
- [U-Net implementation in PyTorch](https://github.com/milesial/Pytorch-UNet)
- [U-Net implementation in TensorFlow](https://github.com/jakeret/tf_unet)
