from keras.layers import Input, SeparableConv2D, BatchNormalization, MaxPooling2D, Conv2DTranspose, Conv2D, Dropout, concatenate
from keras.models import Model
from keras.optimizers import Adam


def mobileunet(inputs, num_classes, drop_prob=0.1, activation='sigmoid'):
    """
    Mobile UNet architecture with separable convolutions and optional dropout.

    Args:
        inputs (tensor): Input tensor, e.g. Input(shape=(H, W, C)).
        num_classes (int): Number of output segmentation classes.
        drop_prob (float): Dropout probability at the bottleneck.
        activation (str): Activation for the final layer.

    Returns:
        keras.Model: Compiled MobileUNet model.
    """
    # Encoder
    conv1 = SeparableConv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = SeparableConv2D(64, 3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = SeparableConv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = SeparableConv2D(128, 3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = SeparableConv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = SeparableConv2D(256, 3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = SeparableConv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = SeparableConv2D(512, 3, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)

    # Bottleneck
    conv5 = SeparableConv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = SeparableConv2D(1024, 3, activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    if drop_prob and drop_prob > 0:
        conv5 = Dropout(drop_prob)(conv5)

    # Decoder
    up6 = Conv2DTranspose(512, 3, strides=(2, 2), padding='same')(conv5)
    merge6 = concatenate([conv4, up6], axis=-1)
    conv6 = SeparableConv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = SeparableConv2D(512, 3, activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2DTranspose(256, 3, strides=(2, 2), padding='same')(conv6)
    merge7 = concatenate([conv3, up7], axis=-1)
    conv7 = SeparableConv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = SeparableConv2D(256, 3, activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2DTranspose(128, 3, strides=(2, 2), padding='same')(conv7)
    merge8 = concatenate([conv2, up8], axis=-1)
    conv8 = SeparableConv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = SeparableConv2D(128, 3, activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2DTranspose(64, 3, strides=(2, 2), padding='same')(conv8)
    merge9 = concatenate([conv1, up9], axis=-1)
    conv9 = SeparableConv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = SeparableConv2D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)

    outputs = Conv2D(num_classes, 1, activation=activation)(conv9)

    # Use `learning_rate` parameter to avoid deprecation warning
    optimizer = Adam(learning_rate=1e-3)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model
