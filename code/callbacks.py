from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint

def setup_callbacks():
    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=0.01,
        patience=5,
        verbose=1,
        mode="auto",
    )
    reduce_lr = ReduceLROnPlateau(
        monitor     = 'val_loss',
        factor      = 0.1,
        patience    = 2,
        verbose     = 0,
        mode        = 'min',
        min_delta   = 0.01,
        min_lr      = 0
    )
    checkpoint = ModelCheckpoint(
        filepath='mnist_dcvae_weights.h5',
        save_best_only=True
    )
    return [early_stop, reduce_lr, checkpoint]