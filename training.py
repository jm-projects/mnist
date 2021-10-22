import os

import pandas as pd
import numpy as np

from tensorflow import keras

from model import creat_model

for dirname, _, filenames in os.walk('data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Data import and adjustment

train = pd.read_csv('data/train.csv')

x_digit = train.drop(['label'], axis = 1) # Select the image data as 'x_axis'
x_digit = np.array(x_digit, dtype=np.float32)/255
x_digit = x_digit.reshape(-1, 28, 28, 1)

y_digit = train['label'] # Select the labels
y_digit = keras.utils.to_categorical(y_digit, num_classes = 10)

''' In some examples I have seen people add additional data to the training dataset here.
I will not do this for the moment for the sake of simplicity, but this is an easy way 
to improve the model'''

x_train = x_digit # Alternatively could concatenate here with more data
y_train = y_digit


# Potentially could introduce data augmentation here for future revisions.

# Train model and save best weights

model = creat_model()
model.summary()

from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5 ,min_lr=0.000001,verbose=1)

from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(filepath='digit-recognizer-model.hdf5',monitor='val_loss',save_best_only=True,save_weights_only=True,verbose=1)

model_hist = model.fit(x = x_train,
                       y = y_train,
                       validation_split=0.25,
                       epochs=10,
                       callbacks=[reduce_lr,checkpoint],
                       verbose=1)
