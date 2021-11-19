import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Model


def creat_model():

    model_activation = tf.nn.relu
    input_shape = (28,28,1)
    input_layer = Input(input_shape)
    layer = Conv2D(32, (5,5),
                   activation=model_activation,
                   padding='same',
                   input_shape=input_shape)(input_layer)
    layer = MaxPool2D((2,2))(layer)
    layer = Conv2D(64,(3,3), activation=tf.nn.relu, padding='same')(layer)
    layer = Conv2D(64,(3,3), activation=tf.nn.relu, padding='same')(layer)
    layer = MaxPool2D((2,2))(layer)
    layer = Conv2D(128,(3,3), activation=tf.nn.relu, padding='same')(layer)
    layer = Conv2D(128,(3,3), activation=tf.nn.relu, padding='same')(layer)
    layer = Conv2D(128,(3,3), activation=tf.nn.relu, padding='same')(layer)
    layer = MaxPool2D((2,2))(layer)

    flatten = Flatten()(layer)

    layer = Dense(512, activation=tf.nn.relu)(flatten)
    layer = Dropout(0.25)(layer)
    layer = Dense(512, activation=tf.nn.relu)(layer)
    layer  = Dropout(0.25)(layer)
    layer = Dense(512, activation=tf.nn.relu)(layer)
    layer  = Dropout(0.25)(layer)
    output_layer = Dense(10, activation=tf.nn.softmax)(layer)
    model = Model(input_layer,output_layer)
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model