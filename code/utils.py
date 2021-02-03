import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from keras.losses import binary_crossentropy

def display_sample_imgs(xtrain, ytrain, gray=False):
    fig, axes = plt.subplots(1, 10, figsize=(20, 10))
    counter = 0
    for i in range(100, 110):
        axes[counter].set_title(ytrain[i])
        if gray:
            axes[counter].imshow(xtrain[i], cmap='gray')
        else:
            axes[counter].imshow(xtrain[i])
        axes[counter].axis('off')
        counter += 1
    fig.tight_layout()
    plt.show()

def plot_train_loss(history):
    plt.figure(figsize=(14,12))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training Loss')
    plt.tight_layout()
    plt.show()

def plot_latent_distribution(encoder, xtrain, ytrain):
    encoded = encoder.predict(xtrain)
    plt.figure(figsize=(14,12))
    plt.scatter(encoded[:,0], encoded[:,1], s=2, c=ytrain, cmap='hsv')
    plt.colorbar()
    plt.grid()
    plt.show()

def generate_latent_imgs(decoder, n=30, figsize=15):
    digit_size = 28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n)) 
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit   
    plt.figure(figsize=(figsize, figsize))
    plt.imshow(figure, cmap='gray')
    plt.axis('off')
    plt.show()