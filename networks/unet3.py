import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dropout
from tensorflow import keras

def ConvBlock(filters):
    def layer(x):
        for _ in range(2):
            x = layers.Conv2D(filters, kernel_size=3, padding='same', kernel_initializer='he_normal', use_bias=True)(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)
        return x
    return layer

def UpsamplingBlock(filters):
    def layer(x):
        x = layers.Conv2DTranspose(filters, (1, 1), strides=(2, 2), kernel_initializer='he_normal')(x)
        x = layers.Conv2D(filters, kernel_size=3, padding='same', kernel_initializer='he_normal', use_bias=True)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        return x
    return layer

def unet(inputs):
    n_classes = 1

    # Encoding
    x1 = ConvBlock(64)(inputs)
    x2 = ConvBlock(128)(layers.MaxPool2D(pool_size=2)(x1))
    x3 = ConvBlock(256)(layers.MaxPool2D(pool_size=2)(x2))
    x4 = ConvBlock(512)(layers.MaxPool2D(pool_size=2)(x3))
    x5 = ConvBlock(1024)(layers.MaxPool2D(pool_size=2)(x4))
    
    # Decoding
    d5 = UpsamplingBlock(512)(x5)
    d5 = layers.Concatenate()([x4, d5])
    d5 = ConvBlock(512)(d5)
    
    d4 = UpsamplingBlock(256)(d5)
    d4 = layers.Concatenate()([x3, d4])
    d4 = ConvBlock(256)(d4)
    
    d3 = UpsamplingBlock(128)(d4)
    d3 = layers.Concatenate()([x2, d3])
    d3 = ConvBlock(128)(d3)
    
    d2 = UpsamplingBlock(64)(d3)
    d2 = layers.Concatenate()([x1, d2])
    d2 = ConvBlock(64)(d2)
    
    outputs = layers.Conv2D(n_classes, kernel_size=1, activation='softmax' if n_classes > 1 else 'sigmoid')(d2)
    
    return keras.Model(inputs, outputs)

