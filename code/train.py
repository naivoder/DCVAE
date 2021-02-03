print('-------------------------------')
print('TRAINING DCVAE ON MNIST DATASET')
print('-------------------------------')

import os
import logging
from build import DCVAE
from build import build_dataset
from callbacks import setup_callbacks
from utils import plot_train_loss
from utils import display_sample_imgs
from utils import generate_latent_imgs
from utils import plot_latent_distribution

if __name__=="__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger('tensorflow').setLevel(logging.FATAL)
    model = DCVAE()
    (xtrain, ytrain), (xtest, ytest), input_shape = build_dataset()
    display_sample_imgs(xtrain, ytrain, gray=True)
    encoder, decoder, vae = model.build_model(input_shape)
    callbacks = setup_callbacks()
    history = vae.fit(xtrain, xtrain, epochs=100, batch_size=32, validation_data=(xtest, xtest), callbacks=callbacks)
    plot_train_loss(history)
    plot_latent_distribution(encoder, xtrain, ytrain)
    generate_latent_imgs(decoder)