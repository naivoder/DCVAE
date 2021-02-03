# Deep Convolutional Variational Autoencoder

This repository contains a Keras implementation using the MNIST digits dataset.  
It improves upon the results demonstrated in the [Keras blog](https://blog.keras.io/building-autoencoders-in-keras.html).  
To achieve better performance, the model employs:

- deeper convolutional architecture
- [batch normalization](http://proceedings.mlr.press/v37/ioffe15.pdf) layers
- [max pooling](https://keras.io/api/layers/pooling_layers/max_pooling2d/) rather than strided convolutions

Latent Space Embedding:  
![Embedded Latent Space](/images/mnist_latent_encoding.png "Latent Space Embedding") 

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

