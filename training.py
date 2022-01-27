import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import creat_model


if __name__ == "__main__":

    """ Data preprocessing:
    Here we simply import the data and scale/reshape it so each
    pixel has a value between 0 and 1, and it is in a 28x28x1
    numpy array"""

    train = pd.read_csv('data/train.csv')

    # Set the image data as 'x_axis'
    x_digit = train.drop(['label'], axis=1)
    x_digit = np.array(x_digit, dtype=np.float32)/255
    x_digit = x_digit.reshape(-1, 28, 28, 1)

    y_digit = train['label']
    y_digit = keras.utils.to_categorical(y_digit, num_classes=10)

    """ Data augmentation:
    Here we use tensorflows ImageDataGenerator to augment the data.
    This is done in batches of 256 for the test data and 64 for
    the validation data"""

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.20,
        shear_range=15,
        zoom_range=0.10,
        validation_split=0.25,
        horizontal_flip=False)

    datagen.fit(x_digit)

    gen_train = datagen.flow(x_digit, y_digit, batch_size=256,
                             subset='training')
    gen_validation = datagen.flow(x_digit, y_digit, batch_size=64,
                                  subset='validation')

    """ Model fit:
    Now we create the model as specified in model.py.
    Checkpoints are used to save the training weights of the best
    fit."""

    model = creat_model()
    model.summary()

    from tensorflow.keras.callbacks import ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=5,
                                  min_lr=0.000001,
                                  verbose=1)

    from tensorflow.keras.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint(
        filepath='digit-recognizer-model.hdf5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True, verbose=1)

    model_hist = model.fit(gen_train,
                           validation_data=gen_validation,
                           epochs=40,
                           callbacks=[reduce_lr, checkpoint],
                           verbose=1)
