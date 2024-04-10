
# Cats vs Dogs Image Classification with VGG16

This repository contains a Python script for image classification of cats and dogs using the VGG16 architecture. The VGG16 model is a popular convolutional neural network (CNN) architecture for image classification tasks.

## Dataset

The code uses the "Dogs vs. Cats" dataset, which can be found on Kaggle. The dataset consists of images of cats and dogs for training and testing the model.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/seyyedmsl82/dogs-vs-cats-with-vgg16.git
   cd dogs-vs-cats-with-vgg16
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle and unzip it:

4. Run the script:

   ```bash
   python 'dogs-vs-cats-with-vgg16 (1).ipynb'
   ```

## Explanation of Code

The script performs the following steps:

1. **Data Preparation:** Downloads and unzips the dataset, then creates DataFrames for training and testing.
2. **Input Processing:** Resizes images to 224x224 pixels and calculates average RGB values for consistent input.
3. **Data Visualization:** Displays sample images from the dataset using matplotlib.
4. **VGG16 Architecture:** Implements VGG16 with additional layers for classification.
5. **Model Training:** Trains the model on the training dataset using TensorFlow and Keras.
6. **Model Evaluation:** Evaluates the model on the validation dataset.
7. **Model Prediction:** Makes predictions on the test dataset.

## VGG16 Architecture

- The VGG16 model is a pre-trained CNN with 16 layers.
- Convolutional layers use 3x3 or 1x1 filters with fixed convolution steps.
- The model can be configured from VGG11 to VGG19 based on the total number of layers.
- VGG11 has 8 convolutional layers and 3 fully connected layers, while VGG19 has 16 convolutional layers and 3 fully connected layers.
- There are 5 pooling layers distributed under different convolutional layers.


![VGG Structure Diagram](https://raw.githubusercontent.com/blurred-machine/Data-Science/master/Deep%20Learning%20SOTA/img/vgg.png)
![VGG Network](https://raw.githubusercontent.com/blurred-machine/Data-Science/master/Deep%20Learning%20SOTA/img/network.png)
![VGG Config](https://raw.githubusercontent.com/blurred-machine/Data-Science/master/Deep%20Learning%20SOTA/img/config3.jpg)


## Results

The model achieves high accuracy on the validation dataset, demonstrating its effectiveness in classifying cat and dog images.

Feel free to modify and experiment with the code for your own projects!
