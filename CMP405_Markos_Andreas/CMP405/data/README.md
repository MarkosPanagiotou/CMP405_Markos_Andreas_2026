# Data

- `mnist_standard.npz`: official MNIST layout used by the project: 60,000 train and 10,000 test images, each 28x28 grayscale.
- `external_digits/`: 20 independent digitally drawn samples (2 per digit), with raw images and MNIST-compatible 28x28 processed versions.

The included NPZ makes the repository runnable offline. `src/prepare_data.py` can rebuild the MNIST file from OpenML if it is removed and internet access is available.
