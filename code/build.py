import numpy as np
from keras import layers
import keras.backend as K
from keras.models import Model
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

def build_dataset():
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    xtrain = np.expand_dims(xtrain/255, -1)
    xtest = np.expand_dims(xtest/255, -1)
    input_shape = xtrain[0].shape
    return (xtrain, ytrain), (xtest, ytest), input_shape



class DCVAE:
    def latent_space(self, inputs):
        self.mu, self.sigma = inputs
        batch = K.shape(self.mu)[0]
        dim = K.int_shape(self.mu)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return self.mu + K.exp(0.5*self.sigma)*epsilon

    def reconstruction_loss(self, true, pred):
        reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred))*np.prod(self.input_shape[:2])
        kl_loss = 1 + self.sigma - K.square(self.mu) - K.exp(self.sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return K.mean(reconstruction_loss + kl_loss)

    def build_model(self, input_shape, ldim=2):
        latent_dim = ldim
        self.input_shape = input_shape
        encoder_input = layers.Input(shape=self.input_shape)
        e1 = layers.Conv2D(8, 3, padding='same', activation='relu')(encoder_input)
        e1 = layers.BatchNormalization()(e1)
        e2 = layers.Conv2D(16, 3, padding='same', activation='relu')(e1)
        e2 = layers.BatchNormalization()(e2)
        e3 = layers.Conv2D(32, 3, padding='same', activation='relu')(e2)
        e3 = layers.BatchNormalization()(e3)
        e3 = layers.MaxPooling2D(2)(e3)
        e4 = layers.Conv2D(32, 3, padding='same', activation='relu')(e3)
        e4 = layers.BatchNormalization()(e4)
        e5 = layers.Conv2D(64, 3, padding='same', activation='relu')(e4)
        e5 = layers.BatchNormalization()(e5)
        e6 = layers.Conv2D(128, 3, padding='same', activation='relu')(e5)
        e6 = layers.BatchNormalization()(e6)
        e6 = layers.MaxPooling2D(2)(e6)
        encoder = layers.Flatten()(e6)
        mu = layers.Dense(latent_dim)(encoder)
        sigma = layers.Dense(latent_dim)(encoder)
        latent_space = layers.Lambda(self.latent_space, output_shape=(latent_dim,))([mu, sigma])
        encoder = Model(encoder_input, latent_space)
        conv_shape = K.int_shape(e6)
        decoder_input = layers.Input(shape=(latent_dim,))
        decoder = layers.Dense(np.prod(conv_shape[1:]), activation='relu')(decoder_input)
        decoder = layers.Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(decoder)
        d1 = layers.Conv2DTranspose(128, 3, padding='same', activation='relu')(decoder)
        d1 = layers.BatchNormalization()(d1)
        d2 = layers.Conv2DTranspose(64, 3, padding='same', activation='relu')(d1)
        d2 = layers.BatchNormalization()(d2)
        d3 = layers.Conv2DTranspose(32, 3, padding='same', activation='relu')(d2)
        d3 = layers.BatchNormalization()(d3)
        d3 = layers.UpSampling2D(2)(d3)
        d4 = layers.Conv2DTranspose(32, 3, padding='same', activation='relu')(d3)
        d4 = layers.BatchNormalization()(d4)
        d5 = layers.Conv2DTranspose(16, 3, padding='same', activation='relu')(d4)
        d5 = layers.BatchNormalization()(d5)
        d6 = layers.Conv2DTranspose(8, 3, padding='same', activation='relu')(d5)
        d6 = layers.BatchNormalization()(d6)
        d6 = layers.UpSampling2D(2)(d6)
        decoder_out = layers.Conv2DTranspose(self.input_shape[-1], kernel_size=3, padding='same', activation='sigmoid')(d6)
        decoder = Model(decoder_input, decoder_out)
        vae = Model(encoder_input, decoder(encoder(encoder_input)))
        opt = Adam(lr=1e-4, amsgrad=True)
        vae = Model(encoder_input, decoder(encoder(encoder_input)))
        vae.compile(optimizer=opt, loss=self.reconstruction_loss)
        return encoder, decoder, vae 

