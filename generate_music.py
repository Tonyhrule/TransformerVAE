import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def generate_music_from_file(encoder, file_path):
    input_data = np.load(file_path)
    input_data = input_data / 128.0  # Normalize if needed
    input_data = np.expand_dims(input_data, axis=0)  # Model expects batch dimension
    latent_representation = encoder.predict(input_data)
    return latent_representation

if __name__ == "__main__":
    # Load the encoder model
    encoder = load_model('vae_encoder.h5')
    generated_data = generate_music_from_file(encoder, 'output/example_piano_roll.npy')
    print("Generated latent representation:", generated_data)
