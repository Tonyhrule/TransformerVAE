import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_transformer_vae(seq_length, num_features, latent_dim):
    # Encoder
    encoder_inputs = layers.Input(shape=(seq_length, num_features))
    x = layers.LayerNormalization()(encoder_inputs)
    x = layers.MultiHeadAttention(num_heads=2, key_dim=latent_dim)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    z = Sampling()([z_mean, z_log_var])

    # Decoder
    decoder_inputs = layers.RepeatVector(seq_length)(z)
    x = layers.LayerNormalization()(decoder_inputs)
    x = layers.MultiHeadAttention(num_heads=2, key_dim=latent_dim)(x, x)
    decoder_outputs = layers.TimeDistributed(layers.Dense(num_features, activation='sigmoid'))(x)

    # Define the model
    vae = models.Model(encoder_inputs, decoder_outputs)
    encoder = models.Model(encoder_inputs, z_mean)

    # Add VAE loss
    kl_loss = -0.5 * tf.reduce_mean(
        z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
    )
    vae.add_loss(kl_loss)
    vae.compile(optimizer='adam', loss='mse')

    return vae, encoder

def load_data(directory):
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
    data = [np.load(f) for f in all_files]
    data = [x / 128.0 for x in data]  # MIDI pitch normalization
    return np.array(data)

def train_model(data_directory, seq_length, num_pitches, latent_dim):
    data = load_data(data_directory)
    train_data = data[:int(len(data)*0.8)]
    val_data = data[int(len(data)*0.8):]

    vae, encoder = build_transformer_vae(seq_length, num_pitches, latent_dim)
    vae.fit(train_data, train_data, epochs=50, batch_size=32, validation_data=(val_data, val_data))
    
    # Save the entire VAE and the encoder model
    vae.save('vae_model.h5')
    encoder.save('vae_encoder.h5')

if __name__ == "__main__":
    train_model('output', 100, 128, 64)  # Example settings
