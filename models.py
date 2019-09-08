import os
from keras import layers
from keras import models
from keras import engine
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dense
from keras.layers import Dropout, Flatten, GRU
from keras.models import Model
from keras.layers.merge import Average, Maximum, Add
from keras.applications.resnet50 import ResNet50

CLASSES = 4
IMG_SIZE = (216,216)
SEQ = 10
BATCH_SIZE = 28

'''
def RNN(in_shape), weights_dir):

    input_img = Input(shape=in_shape)

    x = GRU(128)(input_img)
    x = Dropout(0.9)(x)
    x = Dense(CLASSES, activation='softmax')(x)

    model = Model(inputs=input_img, outputs=x)


    if os.path.exists(weights_dir):
        model.load_weights(weights_dir)


    return model
'''

def finetuned_resnet(include_top, weights_dir, in_shape):

    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=in_shape)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    if include_top:
        x = Dense(CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)
    if os.path.exists(weights_dir):
        model.load_weights(weights_dir, by_name=True)

    return model

def CNN(in_shape, weights_dir):
    #in_shape = (IMG_SIZE[0],IMG_SIZE[1],3)

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

    if os.path.exists(weights_dir):
        model.load_weights(weights_dir)

    return model

def two_stream_model(spatial_weights, temporal_weights):

    spatial_stream = finetuned_resnet(include_top=True, weights_dir=spatial_weights)

    temporal_stream = CNN(input_shape, temporal_weights)

    spatial_output = spatial_stream.output
    temporal_output = temporal_stream.output

    fused_output = Average(name='fusion_layer')([spatial_output, temporal_output])

    model = Model(inputs=[spatial_stream.input, temporal_stream.input],
     outputs=fused_output, name=two_stream)

    return model

if __name__ == '__main__':

    sha = (IMG_SIZE[0],IMG_SIZE[1])
    m = CNN(sha)
    print(m.summary())
