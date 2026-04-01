import tensorflow as tf
from tensorflow.keras import layers, Model

class PhysicalGradientLayer(layers.Layer):
    def __init__(self, delta_z=0.05, **kwargs):
        super(PhysicalGradientLayer, self).__init__(**kwargs)
        self.delta_z = delta_z

    def call(self, inputs):
        # Pad to maintain sequence length during first-order derivative calculation
        x_padded = tf.pad(inputs, [[0, 0], [1, 0], [0, 0]], mode='SYMMETRIC')
        gradient = tf.abs(x_padded[:, 1:, :] - x_padded[:, :-1, :]) / self.delta_z
        return gradient

    def get_config(self):
        config = super(PhysicalGradientLayer, self).get_config()
        config.update({"delta_z": self.delta_z})
        return config

def build_gg_stnet(sequence_length=64, num_features=5, num_classes=7, dropout_rate=0.5):
    inputs = layers.Input(shape=(sequence_length, num_features), name='log_inputs')
    
    # 1. Physical Gradient Extraction
    gradients = PhysicalGradientLayer(delta_z=0.05, name='physical_gradient')(inputs)
    
    # 2. GG-DCNN: Multi-scale spatial feature enhancement
    conv_d1 = layers.Conv1D(8, kernel_size=3, padding='same', dilation_rate=1, activation='relu')(inputs)
    conv_d2 = layers.Conv1D(8, kernel_size=3, padding='same', dilation_rate=2, activation='relu')(inputs)
    conv_d4 = layers.Conv1D(8, kernel_size=3, padding='same', dilation_rate=4, activation='relu')(inputs)
    conv_d8 = layers.Conv1D(8, kernel_size=3, padding='same', dilation_rate=8, activation='relu')(inputs)
    f_multi = layers.Concatenate(name='multi_scale_concat')([conv_d1, conv_d2, conv_d4, conv_d8])
    
    a_space = layers.Conv1D(32, kernel_size=3, padding='same', activation='sigmoid', name='spatial_gate')(gradients)
    
    modulation = layers.Lambda(lambda x: x + 1.0, name='gate_modulation')(a_space)
    f_gg = layers.Multiply(name='gated_fusion')([f_multi, modulation])
    f_gg = layers.Activation('relu')(f_gg)
    
    x_cnn = layers.MaxPooling1D(pool_size=2)(f_gg)
    x_cnn = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x_cnn)
    x_cnn = layers.MaxPooling1D(pool_size=2)(x_cnn)
    x_spatial = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x_cnn)
    
    # 3. Temporal Modeling
    lstm_out = layers.LSTM(128, return_sequences=True, name='temporal_lstm_1')(x_spatial)
    context_vector = layers.LSTM(128, return_sequences=False, name='temporal_lstm_2')(lstm_out)
    
    # 4. Classification & MC Dropout
    # training=True ensures dropout remains active during inference for uncertainty quantification
    x_drop = layers.Dropout(dropout_rate)(context_vector, training=True) 
    outputs = layers.Dense(num_classes, activation='softmax', name='lithology_output')(x_drop)
    
    model = Model(inputs=inputs, outputs=outputs, name='GG_STNet')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

if __name__ == '__main__':
    model = build_gg_stnet()
    model.summary()
    print("[SUCCESS] Model Architecture Built Flawlessly.")