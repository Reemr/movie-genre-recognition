import os
from keras import layers
from keras import models
from keras import engine
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dense
from keras.layers import Dropout, Flatten
from keras.models import Model

CLASSES = 4
IMG_SIZE = (216,216)
SEQ = 10
BATCH_SIZE = 28


def CNN():
    in_shape = (IMG_SIZE[0],IMG_SIZE[1],3)

    input_img = Input(shape=in_shape)

    x = Conv2D(96,(7,7), strides=(2,2), padding='same', activation='relu')(input_img)
    x = layers.BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(256,(5,5), strides=(2,2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(512,(3,3), strides=(1,1), padding='same', activation='relu')(x)

    x = Conv2D(512,(3,3), strides=(1,1), padding='same', activation='relu')(x)

    x = Conv2D(256,(3,3), strides=(1,1), padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2))(x)

    x = layers.Flatten()(x)

    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(CLASSES, activation='softmax')(x)

    model = Model(inputs=input_img, outputs=x)

    return model


if __name__ == '__main__':
    Model = CNN()
    print(Model.summary())
