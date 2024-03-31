# DeepDream Image Generation

This Python script uses TensorFlow and Keras to generate DeepDream images. DeepDream is an algorithm that enhances patterns in images by iteratively applying them at different scales. This implementation uses the InceptionV3 model for feature extraction.

## Requirements

- Python
- TensorFlow
- Keras
- NumPy

Install the required packages using:

```bash
pip install tensorflow keras numpy
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/seyyedmsl82/Deep-Dream.git
cd Deep-Dream
```

2. Run the script:

```bash
python Deep_Dream_using_TensorFlow.ipynb
```

## Configuration

You can customize the following parameters in the script:

- `image_path`: URL of the input image
- `layer_settings`: Dictionary of layer names and their importance factors
- `lr`: Learning rate for gradient ascent
- `num_octave`: Number of octaves to generate
- `octave_scale`: Scale factor between octaves
- `iteration`: Number of iterations per octave
- `max_loss`: Maximum loss allowed before stopping

Adjust these parameters to experiment with different DeepDream effects.

## Example

![Generated Image](image.png)

This image was generated using the provided example script with the specified parameters. Feel free to use your own images and customize the script for more creative results.
