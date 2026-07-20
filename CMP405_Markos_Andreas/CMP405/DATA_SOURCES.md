# Data sources and provenance

The experiments use the MNIST handwritten digit dataset (70,000 grayscale 28x28 digit images, with the standard 60,000 train / 10,000 test split).

For the executed experiments, the standard split was reconstructed from the following public CSV mirror:

- `https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/mnist_train.csv`
- `https://raw.githubusercontent.com/ben519/nnets-for-your-dog/master/data/mnist_test.csv`

That mirror stores 50,000 examples in the first file and 20,000 in the second. The project reconstructs the standard split as 50,000 + the first 10,000 examples of the second file for training, and the final 10,000 examples for the standard test set. A consistency check verified the test split using its known initial labels (`7, 2, 1, 0, 4, ...`), its 10,000-example size, and class counts.

The resulting data is stored locally as `data/mnist_standard.npz`, so the delivered project runs offline.

For reproducibility when the local NPZ is absent, `src/prepare_data.py` can obtain MNIST through:

```python
sklearn.datasets.fetch_openml("mnist_784", version=1)
```

The 20 A2 images are independent digitally drawn vector-stroke samples generated specifically for this project. They do not reuse MNIST train/test images.
