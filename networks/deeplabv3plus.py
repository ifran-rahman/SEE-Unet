# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# A
import tensorflow as tf
from tensorflow.keras import layers, Model

def DeepLabV3Plus(
    input_tensor,
    num_classes=1,
    output_stride=16,
    activation='sigmoid'
):
    """
    DeepLabV3+ encoder-decoder with ResNet50 backbone.

    Args:
      input_tensor: tf.keras.Input or tensor, shape (H, W, C)
      num_classes: int, number of output channels
      output_stride: 8 or 16, controls dilation in backbone
      activation: final activation (e.g. 'sigmoid' or None for logits)
    Returns:
      tf.keras.Model
    """
    # -----------------------------------------
    # Backbone: ResNet50
    # -----------------------------------------
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=input_tensor
    )

    # Only modify conv5 for OS=16
    if output_stride == 16:
        for layer in base_model.layers:
            if isinstance(layer, layers.Conv2D) and layer.name.startswith('conv5_'):
                # 1) remove down-sampling if present
                if hasattr(layer, 'strides') and (layer.strides[0] > 1 or layer.strides[1] > 1):
                    layer.strides = (1, 1)
                # 2) apply atrous rate = 2
                layer.dilation_rate = (2, 2)

    elif output_stride == 8:
        # if you ever need OS=8, you’d dilate both conv4 & conv5 similarly
        raise ValueError("output_stride=8 not implemented")

    else:
        raise ValueError("output_stride must be 8 or 16")

    # -----------------------------------------
    # Feature maps
    # -----------------------------------------
    high_level = base_model.get_layer('conv4_block6_2_relu').output  # OS originally 16
    low_level  = base_model.get_layer('conv2_block3_2_relu').output  # OS = 4

    # -----------------------------------------
    # Atrous Spatial Pyramid Pooling (ASPP)
    # -----------------------------------------
    def _aspp(x):
        # 1×1 conv branch
        b0 = layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
        b0 = layers.BatchNormalization()(b0)
        b0 = layers.ReLU()(b0)

        # dilated conv branches
        rates = [6, 12, 18]
        bs = []
        for r in rates:
            b = layers.Conv2D(256, 3, padding='same', dilation_rate=r, use_bias=False)(x)
            b = layers.BatchNormalization()(b)
            b = layers.ReLU()(b)
            bs.append(b)

        # image pooling branch
        img_pool = layers.GlobalAveragePooling2D()(x)
        img_pool = layers.Reshape((1, 1, -1))(img_pool)
        img_pool = layers.Conv2D(256, 1, padding='same', use_bias=False)(img_pool)
        img_pool = layers.BatchNormalization()(img_pool)
        img_pool = layers.ReLU()(img_pool)
        img_pool = layers.Resizing(
            height=x.shape[1],
            width=x.shape[2],
            interpolation="bilinear"
        )(img_pool)
                # concatenate and project
        x = layers.Concatenate()([b0, *bs, img_pool])
        x = layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    aspp_out = _aspp(high_level)

    # -----------------------------------------
    # Decoder
    # -----------------------------------------
    # compress low-level features
    ll = layers.Conv2D(48, 1, padding='same', use_bias=False)(low_level)
    ll = layers.BatchNormalization()(ll)
    ll = layers.ReLU()(ll)

    # upsample ASPP to low-level size
    aspp_up = layers.Resizing(
        height=ll.shape[1],
        width=ll.shape[2],
        interpolation="bilinear"
    )(aspp_out)

    x = layers.Concatenate()([aspp_up, ll])

    # two depthwise separable convs
    x = layers.SeparableConv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # final upsample to input resolution
# final upsample to input resolution
    x = layers.Resizing(
        height=input_tensor.shape[1],
        width=input_tensor.shape[2],
        interpolation="bilinear"
    )(x)

    # output logits or [0,1] probabilities
    outputs = layers.Conv2D(num_classes, 1, padding='same', activation=activation)(x)

    return Model(inputs=input_tensor, outputs=outputs)

# B
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import tensorflow as tf
# from tensorflow.keras import layers, Model


# def _resize_like_output_shape(input_shapes):
#     """
#     Shared output_shape function for every "resize tensor A to match
#     the spatial size of tensor B" Lambda in this model.

#     input_shapes is the list [shape_of_tensor_being_resized, shape_of_target_tensor].
#     Output keeps the batch dim + channel dim of the tensor being resized,
#     but adopts the height/width of the target tensor.
#     """
#     resized_shape, target_shape = input_shapes
#     return (resized_shape[0], target_shape[1], target_shape[2], resized_shape[-1])


# def DeepLabV3Plus(
#     input_tensor,
#     num_classes=1,
#     output_stride=16,
#     activation='sigmoid'
# ):
#     """
#     DeepLabV3+ encoder-decoder with ResNet50 backbone.

#     Args:
#       input_tensor: tf.keras.Input or tensor, shape (H, W, C)
#       num_classes: int, number of output channels
#       output_stride: 8 or 16, controls dilation in backbone
#       activation: final activation (e.g. 'sigmoid' or None for logits)
#     Returns:
#       tf.keras.Model
#     """
#     # -----------------------------------------
#     # Backbone: ResNet50
#     # -----------------------------------------
#     base_model = tf.keras.applications.ResNet50(
#         weights='imagenet',
#         include_top=False,
#         input_tensor=input_tensor
#     )

#     # Only modify conv5 for OS=16
#     if output_stride == 16:
#         for layer in base_model.layers:
#             if isinstance(layer, layers.Conv2D) and layer.name.startswith('conv5_'):
#                 # 1) remove down-sampling if present
#                 if hasattr(layer, 'strides') and (layer.strides[0] > 1 or layer.strides[1] > 1):
#                     layer.strides = (1, 1)
#                 # 2) apply atrous rate = 2
#                 layer.dilation_rate = (2, 2)

#     elif output_stride == 8:
#         # if you ever need OS=8, you'd dilate both conv4 & conv5 similarly
#         raise ValueError("output_stride=8 not implemented")

#     else:
#         raise ValueError("output_stride must be 8 or 16")

#     # -----------------------------------------
#     # Feature maps
#     # -----------------------------------------
#     high_level = base_model.get_layer('conv4_block6_2_relu').output  # OS originally 16
#     low_level  = base_model.get_layer('conv2_block3_2_relu').output  # OS = 4

#     # -----------------------------------------
#     # Atrous Spatial Pyramid Pooling (ASPP)
#     # -----------------------------------------
#     def _aspp(x):
#         # 1x1 conv branch
#         b0 = layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
#         b0 = layers.BatchNormalization()(b0)
#         b0 = layers.ReLU()(b0)

#         # dilated conv branches
#         rates = [6, 12, 18]
#         bs = []
#         for r in rates:
#             b = layers.Conv2D(256, 3, padding='same', dilation_rate=r, use_bias=False)(x)
#             b = layers.BatchNormalization()(b)
#             b = layers.ReLU()(b)
#             bs.append(b)

#         # image pooling branch
#         img_pool = layers.GlobalAveragePooling2D()(x)
#         img_pool = layers.Reshape((1, 1, -1))(img_pool)
#         img_pool = layers.Conv2D(256, 1, padding='same', use_bias=False)(img_pool)
#         img_pool = layers.BatchNormalization()(img_pool)
#         img_pool = layers.ReLU()(img_pool)
#         img_pool = layers.Lambda(
#             lambda tensors: tf.image.resize(
#                 tensors[0],
#                 tf.shape(tensors[1])[1:3],
#                 method='bilinear'
#             ),
#             output_shape=_resize_like_output_shape
#         )([img_pool, x])

#         # concatenate and project
#         x = layers.Concatenate()([b0, *bs, img_pool])
#         x = layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.ReLU()(x)
#         return x

#     aspp_out = _aspp(high_level)

#     # -----------------------------------------
#     # Decoder
#     # -----------------------------------------
#     # compress low-level features
#     ll = layers.Conv2D(48, 1, padding='same', use_bias=False)(low_level)
#     ll = layers.BatchNormalization()(ll)
#     ll = layers.ReLU()(ll)

#     # upsample ASPP to low-level size
#     aspp_up = layers.Lambda(
#         lambda tensors: tf.image.resize(
#             tensors[0],
#             tf.shape(tensors[1])[1:3],
#             method='bilinear'
#         ),
#         output_shape=_resize_like_output_shape
#     )([aspp_out, ll])

#     x = layers.Concatenate()([aspp_up, ll])

#     # two depthwise separable convs
#     x = layers.SeparableConv2D(256, 3, padding='same', use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)

#     x = layers.SeparableConv2D(256, 3, padding='same', use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)

#     # final upsample to input resolution
#     x = layers.Lambda(
#         lambda tensors: tf.image.resize(
#             tensors[0],
#             tf.shape(tensors[1])[1:3],
#             method='bilinear'
#         ),
#         output_shape=_resize_like_output_shape
#     )([x, input_tensor])

#     # output logits or [0,1] probabilities
#     outputs = layers.Conv2D(num_classes, 1, padding='same', activation=activation)(x)

#     return Model(inputs=input_tensor, outputs=outputs)