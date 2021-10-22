import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Model

from model import creat_model


model = creat_model()
model.load_weights('digit-recognizer-model.hdf5')

test = pd.read_csv('data/test.csv')
test = np.array(test, dtype=np.float32)/255
test = test.reshape(-1,28,28,1)

prediction = model.predict(test)

predict = np.array(np.round(prediction), dtype = np.int32)
predict = np.argmax(predict , axis=1).reshape(-1, 1)
out = [{'ImageId': i+1, 'Label': predict[i][0]} for i in range(len(predict))]
pd.DataFrame(out).to_csv('submission.csv', index=False)

print(len(out))