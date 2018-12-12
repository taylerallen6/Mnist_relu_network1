# Mnist_relu_network1

This is an example of a fully connected neural network classifying the mnist dataset.

The network is written in python3 using <a href="https://github.com/pytorch">Pytorch</a> for tensor operations. It has 3 layers with a relu activation function in the hidden layer and a log softmax activation function at the output layer.

Batch size = 1
Input size = 784 (28 x 28 for each image)
Hidden size = 200
output size = 10 (value for each digit)

```
BATCH_SIZE = 1
N, D_in, H, D_out = BATCH_SIZE, 784, 200, 10
n_epochs = 20
```
