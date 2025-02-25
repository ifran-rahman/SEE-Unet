#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.layers import concatenate, Conv2DTranspose, BatchNormalization, SeparableConv2D, Activation, Add
from keras import backend as K

from keras.layers import Conv2D, BatchNormalization, SeparableConv2D, concatenate, GlobalAveragePooling2D, Dense, Reshape, multiply
from keras import backend as K

def squeeze_and_excite(x, filters, se_ratio=16):
    """ Squeeze-and-Excitation Block """
    
    se = GlobalAveragePooling2D()(x)
    se = Dense(filters * 2 // se_ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)

    # Reshape for channel-wise scaling
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    se_shape = [1, 1, filters] if channel_axis == -1 else [filters, 1, 1]
    se = Reshape(se_shape)(se)

    # Scale input features
    x = multiply([x, se])  
    
    return x

def SEE_Block(x, see_id, squeeze=16, expand=64, se_ratio=16):
    """ SEE Block with Unique Layer Naming """
    
    # Generate unique name for each layer
    def unique_layer_name(name):
        return f"see{see_id}_{name}"

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # Squeeze Step
    squeezed = Conv2D(squeeze, (1, 1), activation='relu', padding='same', name=unique_layer_name("squeeze1x1"))(x)
    squeezed = BatchNormalization(axis=channel_axis, name=unique_layer_name("batchnorm"))(squeezed)

    # Expand Step
    expanded = SeparableConv2D(expand, (3, 3), activation='relu', padding='same', name=unique_layer_name("expand3x3"))(squeezed)

    # Apply Squeeze-and-Excitation Block
    x = squeeze_and_excite(expanded, expand)
    
    return x


def SEE_Unet(inputs, num_classes=1, deconv_ksize=3, dropout=0.5, activation='sigmoid'):
    """
    :param inputs: input layer.
    :param num_classes: number of classes.
    :param deconv_ksize: (width and height) or integer of the 2D deconvolution window.
    :param dropout: dropout rate
    :param activation: type of activation at the top layer.
    :returns: SqueezeUNet model
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    if num_classes is None:
        num_classes = K.int_shape(inputs)[channel_axis]

    x01 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu', name='conv1')(inputs)
    x02 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1', padding='same')(x01)

    # DS1
    x03 = SEE_Block(x02, see_id=2, squeeze=16, expand=64)
    x04 = SEE_Block(x03, see_id=3, squeeze=16, expand=64)
    x05 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3', padding="same")(x04)

    # DS2    
    x06 = SEE_Block(x05, see_id=4, squeeze=32, expand=128)
    x07 = SEE_Block(x06, see_id=5, squeeze=32, expand=128)
    x08 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5', padding="same")(x07)

    # DS3
    x09 = SEE_Block(x08, see_id=6, squeeze=48, expand=192)
    x10 = SEE_Block(x09, see_id=7, squeeze=48, expand=192)

    # DS4
    x11 = SEE_Block(x10, see_id=8, squeeze=64, expand=256)
    x12 = SEE_Block(x11, see_id=9, squeeze=64, expand=256)

    if dropout != 0.0:
        x12 = Dropout(dropout)(x12)

    # US1
    up1 = concatenate([
        Conv2DTranspose(192, deconv_ksize, strides=(1, 1), padding='same')(x12),
        x10,
    ], axis=channel_axis)
    up1 = SEE_Block(up1, see_id=10, squeeze=48, expand=192)

    # US2
    up2 = concatenate([
        Conv2DTranspose(128, deconv_ksize, strides=(1, 1), padding='same')(up1),
        x08,
    ], axis=channel_axis)
    up2 = SEE_Block(up2, see_id=11, squeeze=32, expand=128)

    # US3
    up3 = concatenate([
        Conv2DTranspose(64, deconv_ksize, strides=(2, 2), padding='same')(up2),
        x05,
    ], axis=channel_axis)
    up3 = SEE_Block(up3, see_id=12, squeeze=16, expand=64)

    # US4
    up4 = concatenate([
        Conv2DTranspose(32, deconv_ksize, strides=(2, 2), padding='same')(up3),
        x02,
    ], axis=channel_axis)
    up4 = SEE_Block(up4, see_id=13, squeeze=16, expand=32)
    up4 = UpSampling2D(size=(2, 2))(up4)
    
    x = concatenate([up4, x01], axis=channel_axis)
    
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(num_classes, (1, 1), activation=activation)(x)

    return Model(inputs=inputs, outputs=x)
