# Deep Convolutional Variational Autoencoder

This repository contains a Keras implementation using the MNIST digits dataset.  
It improves upon the results demonstrated in the [Keras blog](https://blog.keras.io/building-autoencoders-in-keras.html).  
To achieve better performance, the model makes use of a deeper convolutional architecture as well as [batch normalization](http://proceedings.mlr.press/v37/ioffe15.pdf) layers and max-pooling rather than strided convolutions.

![Embedded Latent Space](/images/mnist_latent_encoding.png "Latent Space Embedding") ![Generated Digits](/images/mnist_generated_digits.png "Generated Digits")

---

## Installation

A requirements.txt file has been provided.  
Once the repository has been cloned, create a new conda environment:

```bash
cd DeepConvolutionalVariationalAutoencoder
conda create env --name DCVAE --file requirements.txt
conda activate DCVAE
```

---

## Usage

Run training from the command line:  

```bash
cd code  
python train.py
```

Run training with jupyter notebook:

```bash
cd notebook
jupyter notebook
```

