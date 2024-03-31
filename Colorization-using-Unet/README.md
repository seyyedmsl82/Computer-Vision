# UNet Colorization: Image Colorization Using PyTorch

## Overview

This repository presents a PyTorch implementation of a UNet-based image colorization model. Image colorization is the process of adding color to black-and-white or grayscale images, and UNet architectures have proven effective for various image-to-image translation tasks. This project aims to provide a comprehensive explanation of the implementation, usage, and training process for the UNet colorization model.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Colorization Model](#colorization-model)
- [Understanding UNet: A Fundamental Overview](#understanding-unet-a-fundamental-overview)
- [Applications of UNet](#Applications-of-UNet)
- [Results](#results)
- [Customization](#customization)

## Introduction

Image colorization is a fascinating computer vision task that involves transforming grayscale images into their corresponding colored versions. The motivation behind this project is to leverage the power of UNet architecture for accurate and detailed colorization. UNet, originally designed for biomedical image segmentation, has shown versatility in various image-related tasks.

## Requirements

Before diving into the code, ensure you have the necessary packages installed. The primary dependency is the `torch_snippets` package. Install it using the following command:

```bash
pip install torch_snippets
```

Additionally, make sure you have the required PyTorch and other dependencies installed.

## Usage

To use the UNet colorization model, follow these steps:

1. **Install Dependencies:**
   
    ```bash
    pip install torch_snippets
    ```

2. **Download CIFAR-10 Dataset:**
   
    ```python
    from torchvision import datasets
    import torch
    data_folder = '~/cifar10/cifar/'
    datasets.CIFAR10(data_folder, download=True)
    ```

3. **Run the Code:**
   
    Copy and paste the provided code into your Python environment.

## Colorization Model

The core of this repository is the `UNet` model, a convolutional neural network architecture. The UNet model consists of DownConv and UpConv modules, which are responsible for downscaling and upscaling, respectively. The final model is designed to transform grayscale images into their colored counterparts.

```python
# UNet model architecture
class UNet(nn.Module):
  def __init__(self):
    # ... (implementation details)
    
  def forward(self, x):
    # ... (forward pass details)
    return x
```

## Understanding UNet: A Fundamental Overview

UNet, introduced by Olaf Ronneberger, Philipp Fischer, and Thomas Brox in 2015, is a convolutional neural network architecture designed for semantic segmentation tasks. Its distinctive U-shaped architecture has made it a popular choice in various image-to-image translation tasks, including medical image segmentation, image colorization, and more. This overview aims to provide a fundamental understanding of the UNet architecture, its components, and its applications.

### Anatomy of UNet

1. Encoder (Contracting Path)

The UNet architecture is divided into two main parts: the encoder (contracting path) and the decoder (expansive path).

- **Convolutional Blocks:**
  The encoder consists of a series of convolutional blocks, each containing two 3x3 convolutions, followed by batch normalization and a rectified linear unit (ReLU) activation function. These blocks are responsible for capturing hierarchical features from the input image.

- **Max Pooling:**
  After each convolutional block, a max-pooling operation is applied to reduce the spatial dimensions of the features, helping the model focus on more abstract and high-level features.

2. Bottleneck

The bottleneck serves as the bridge between the encoder and decoder. It is a stack of convolutional layers that retains the most essential information from the input image.

3. Decoder (Expansive Path)

The decoder is the symmetric counterpart to the encoder, aiming to reconstruct the spatial resolution of the input image.

- **Upconvolutional Blocks:**
  Each block in the decoder consists of an upconvolutional layer (transposed convolution or upsampled convolution), followed by two 3x3 convolutions, batch normalization, and ReLU activation. These blocks help increase the spatial resolution of the features.

- **Skip Connections:**
  One of the key features of UNet is the use of skip connections. The feature maps from the encoder are concatenated with the corresponding feature maps in the decoder. This enables the network to utilize both low-level and high-level features during the reconstruction process.

4. Output Layer

The final layer of the decoder typically consists of a 1x1 convolutional layer, which reduces the number of channels to match the desired output.


![UNet-structure](https://github.com/seyyedmsl82/Colorization-using-Unet/blob/main/The-architecture-of-Unet.png)


## Applications of UNet

1. Semantic Segmentation

UNet's original purpose was semantic segmentation, where it excels in delineating and classifying objects within an image. The skip connections help maintain fine-grained details during segmentation.

2. Medical Image Segmentation

UNet has been widely adopted in medical image analysis, particularly for tasks such as tumor segmentation, organ detection, and cell segmentation. Its ability to handle limited annotated data makes it suitable for medical applications.

3. Image-to-Image Translation

Due to its architecture's versatility, UNet has been applied to various image translation tasks. In image colorization, for example, a UNet model can be trained to transform grayscale images into colored versions.

## Results

The training and validation logs are plotted to visualize the model's performance over epochs. Additionally, sample visualizations of predictions are included to showcase the colorization results.

## Customization

This project is designed for flexibility and experimentation. Users can customize various aspects such as the dataset, batch sizes, and model parameters to suit their specific colorization tasks. Adjust the data folder path, batch sizes, and other parameters based on your specific use case.
