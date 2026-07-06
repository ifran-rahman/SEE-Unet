
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

# --- Custom Layers with get_config for serialization ---

class StochasticDepth(layers.Layer):
    """Implements DropPath (a.k.a. Stochastic Depth)."""
    def __init__(self, drop_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if (not training) or (self.drop_prob == 0.0):
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
        binary_tensor = tf.floor(random_tensor)
        return tf.divide(x, keep_prob) * binary_tensor

    def get_config(self):
        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})
        return config

class EfficientSelfAttention(layers.Layer):
    """Factorized self-attention with spatial reduction and stochastic depth."""
    def __init__(self, dim, num_heads, reduction_ratio, drop_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.red_ratio = reduction_ratio
        self.q_proj = layers.Dense(dim)
        self.kv_conv = layers.Conv2D(2 * dim,
                                     kernel_size=reduction_ratio,
                                     strides=reduction_ratio,
                                     padding='valid')
        self.out_proj = layers.Dense(dim)
        self.drop_path = StochasticDepth(drop_prob)

    def call(self, x, training=None):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        q = self.q_proj(x)
        kv = self.kv_conv(x)
        k, v = tf.split(kv, num_or_size_splits=2, axis=-1)

        q = tf.reshape(q, (B, H * W, self.num_heads, C // self.num_heads))
        k = tf.reshape(k, (B, (H // self.red_ratio) * (W // self.red_ratio),
                           self.num_heads, C // self.num_heads))
        v = tf.reshape(v, (B, (H // self.red_ratio) * (W // self.red_ratio),
                           self.num_heads, C // self.num_heads))

        scale = tf.cast(C // self.num_heads, tf.float32) ** -0.5
        attn = tf.einsum('bqhd,bkhd->bhqk', q, k) * scale
        attn = tf.nn.softmax(attn, axis=-1)

        out = tf.einsum('bhqk,bkhd->bqhd', attn, v)
        out = tf.reshape(out, (B, H, W, C))
        out = self.out_proj(out)
        return self.drop_path(out, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "reduction_ratio": self.red_ratio,
            "drop_prob": self.drop_path.drop_prob,
        })
        return config

class MixFFN(layers.Layer):
    """Feed-forward network with expansion + depthwise convolution + stochastic depth."""
    def __init__(self, dim, expansion, drop_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.expansion = expansion
        self.expanded_dim = int(dim * expansion)
        self.fc1 = layers.Dense(self.expanded_dim)
        self.dwconv = layers.DepthwiseConv2D(
            kernel_size=3,
            padding='same',
            depth_multiplier=1
        )
        self.act = layers.Activation('gelu')
        self.fc2 = layers.Dense(dim)
        self.drop_path = StochasticDepth(drop_prob)

    def call(self, x, training=None):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.drop_path(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "expansion": self.expansion,
            "drop_prob": self.drop_path.drop_prob,
        })
        return config

# --- SegFormer-B0 Model Builder ---

def SegFormer(
    input_tensor,
    num_classes=1,
    dims=(64, 128, 320, 512),
    heads=(1, 2, 5, 8),
    ff_expansions=(8, 8, 4, 4),
    reduction_ratios=(8, 4, 2, 1),
    num_blocks=(2, 2, 2, 2),
    drop_prob=0.1,
    activation=None
):
    """Builds SegFormer-B0 in TensorFlow."""
    x = input_tensor
    features = []
    stage_cfgs = [
        {'stride': 4, 'kernel': 7, 'pad': 3},
        {'stride': 2, 'kernel': 3, 'pad': 1},
        {'stride': 2, 'kernel': 3, 'pad': 1},
        {'stride': 2, 'kernel': 3, 'pad': 1},
    ]

    for i in range(4):
        cfg = stage_cfgs[i]
        x = layers.ZeroPadding2D(cfg['pad'])(x)
        x = layers.Conv2D(dims[i], cfg['kernel'], strides=cfg['stride'], padding='valid')(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        for _ in range(num_blocks[i]):
            attn = layers.LayerNormalization(epsilon=1e-6)(x)
            attn = EfficientSelfAttention(
                dim=dims[i],
                num_heads=heads[i],
                reduction_ratio=reduction_ratios[i],
                drop_prob=drop_prob
            )(attn)
            x = layers.Add()([x, attn])

            ffn = layers.LayerNormalization(epsilon=1e-6)(x)
            ffn = MixFFN(
                dim=dims[i],
                expansion=ff_expansions[i],
                drop_prob=drop_prob
            )(ffn)
            x = layers.Add()([x, ffn])

        features.append(x)

    # Decoder
    up_feats = []
    for i, feat in enumerate(features):
        scale = 2 ** i
        u = layers.UpSampling2D(size=scale, interpolation='bilinear')(feat)
        u = layers.Conv2D(256, 1)(u)
        up_feats.append(u)

    x = layers.Concatenate()(up_feats)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(drop_prob)(x)
    x = layers.Conv2D(num_classes, 1, activation=activation)(x)
    outputs = layers.UpSampling2D(size=4, interpolation='bilinear')(x)

    return Model(inputs=input_tensor, outputs=outputs, name='SegFormer')
