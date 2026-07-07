import tensorflow as tf
from tensorflow.keras import layers, Model


class LogGradientLayer(layers.Layer):
    """First-order log-gradient extraction on the native depth grid."""
    def __init__(self, delta_z=0.125, **kwargs):
        super().__init__(**kwargs)
        self.delta_z = float(delta_z)

    def call(self, inputs):
        # Keep sequence length unchanged
        x_padded = tf.pad(inputs, [[0, 0], [1, 0], [0, 0]], mode="SYMMETRIC")
        gradient = tf.abs(x_padded[:, 1:, :] - x_padded[:, :-1, :]) / self.delta_z
        return gradient

    def get_config(self):
        config = super().get_config()
        config.update({"delta_z": self.delta_z})
        return config


class GradientGatedDepthContext(layers.Layer):
    """
    Gradient-gated depth-context modeling:
    h_t = T_t ⊙ h~_t + (1 - T_t) ⊙ h_{t-1}
    """
    def __init__(self, hidden_units=128, return_sequences=True, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.return_sequences = return_sequences

        self.gate_dense = layers.Dense(hidden_units, activation="sigmoid", name=f"{kwargs.get('name', 'ggdc')}_gate")
        self.x_dense = layers.Dense(hidden_units, use_bias=False, name=f"{kwargs.get('name', 'ggdc')}_x")
        self.h_dense = layers.Dense(hidden_units, use_bias=False, name=f"{kwargs.get('name', 'ggdc')}_h")
        self.norm = layers.LayerNormalization()

    def call(self, inputs):
        x_seq, g_seq = inputs  # [B, T, F], [B, T, G]
        # Input normalization to stabilize training
        x_seq = layers.LayerNormalization()(x_seq)
        g_seq = layers.LayerNormalization()(g_seq)

        x_tm = tf.transpose(x_seq, [1, 0, 2])  # [T, B, F]
        g_tm = tf.transpose(g_seq, [1, 0, 2])  # [T, B, G]

        batch_size = tf.shape(x_seq)[0]
        init_h = tf.zeros((batch_size, self.hidden_units), dtype=x_seq.dtype)

        def step(prev_h, elems):
            x_t, g_t = elems
            gate_t = self.gate_dense(g_t)
            cand_t = tf.nn.tanh(self.x_dense(x_t) + self.h_dense(prev_h))
            h_t = gate_t * cand_t + (1.0 - gate_t) * prev_h
            h_t = self.norm(h_t)
            return h_t

        h_tm = tf.scan(
            fn=step,
            elems=(x_tm, g_tm),
            initializer=init_h
        )  # [T, B, H]

        h_seq = tf.transpose(h_tm, [1, 0, 2])  # [B, T, H]

        if self.return_sequences:
            return h_seq
        return h_seq[:, -1, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "return_sequences": self.return_sequences
        })
        return config


def build_gg_stnet(sequence_length=32, num_features=5, num_classes=7):
    reg = tf.keras.regularizers.l2(1e-4)
    inputs = layers.Input(shape=(sequence_length, num_features), name="log_inputs")

    # 1) First-order log-gradient cue
    gradients = LogGradientLayer(delta_z=0.125, name="log_gradient")(inputs)

    # 2) GG-DCNN: multi-scale dilated convolution + gradient-guided spatial gate (add residual & L2 reg)
    conv_d1 = layers.Conv1D(16, 3, padding="same", dilation_rate=1, activation="relu", kernel_regularizer=reg, name="dilated_conv_1")(inputs)
    conv_d2 = layers.Conv1D(16, 3, padding="same", dilation_rate=2, activation="relu", kernel_regularizer=reg, name="dilated_conv_2")(inputs)
    conv_d4 = layers.Conv1D(16, 3, padding="same", dilation_rate=4, activation="relu", kernel_regularizer=reg, name="dilated_conv_4")(inputs)
    conv_d8 = layers.Conv1D(16, 3, padding="same", dilation_rate=8, activation="relu", kernel_regularizer=reg, name="dilated_conv_8")(inputs)

    f_multi = layers.Concatenate(name="multi_scale_concat")([conv_d1, conv_d2, conv_d4, conv_d8])
    # Residual branch
    res_branch = layers.Conv1D(64, 1, padding="same", activation=None, kernel_regularizer=reg)(inputs)
    f_multi = layers.Add()([f_multi, res_branch])
    f_multi = layers.ReLU()(f_multi)

    a_space = layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="sigmoid",
        kernel_regularizer=reg,
        name="spatial_gate"
    )(gradients)

    gate_scale = layers.Lambda(lambda x: 1.0 + x, name="gate_residual")(a_space)
    f_gg = layers.Multiply(name="gradient_guided_fusion")([f_multi, gate_scale])

    # three convolutional blocks with dropout regularization
    x = layers.Conv1D(32, 3, padding="same", activation="relu", kernel_regularizer=reg, name="conv_32")(f_gg)
    x = layers.MaxPooling1D(pool_size=2, strides=2, name="pool_1")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(64, 3, padding="same", activation="relu", kernel_regularizer=reg, name="conv_64")(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2, name="pool_2")(x)
    x = layers.Dropout(0.2)(x)

    x_spatial = layers.Conv1D(128, 3, padding="same", activation="relu", kernel_regularizer=reg, name="conv_128")(x)
    x_spatial = layers.Dropout(0.3)(x_spatial)

    # Dynamic downsample scale, compatible with arbitrary window size
    down_scale = 4
    g_context = layers.AveragePooling1D(pool_size=down_scale, strides=down_scale, padding="same", name="gradient_downsample")(gradients)

    # 3) Two layers Gradient-gated depth-context modeling
    h_seq = GradientGatedDepthContext(
        hidden_units=128,
        return_sequences=True,
        name="depth_context_1"
    )([x_spatial, g_context])

    # Second GGDC use self temporal feature as gate input to avoid redundant gradient
    h_gate_aux = layers.Conv1D(128, 1, padding="same", kernel_regularizer=reg)(h_seq)
    h_vec = GradientGatedDepthContext(
        hidden_units=128,
        return_sequences=False,
        name="depth_context_2"
    )([h_seq, h_gate_aux])

    # 4) Feature fusion + dropout + softmax output
    spatial_vec = layers.GlobalAveragePooling1D(name="spatial_pool")(x_spatial)
    fused = layers.Concatenate(name="feature_fusion")([spatial_vec, h_vec])
    fused = layers.Dropout(0.4)(fused)

    outputs = layers.Dense(num_classes, activation="softmax", kernel_regularizer=reg, name="lithology_output")(fused)

    model = Model(inputs=inputs, outputs=outputs, name="GG_STNet")
    return model


# Custom layer dict for model load/save serialization
CUSTOM_LAYERS = {
    "LogGradientLayer": LogGradientLayer,
    "GradientGatedDepthContext": GradientGatedDepthContext
}


if __name__ == "__main__":
    model = build_gg_stnet(sequence_length=32, num_features=5, num_classes=7)
    model.summary()
    print("[SUCCESS] GG-STNet architecture built successfully.")
