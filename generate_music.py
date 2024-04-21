import torch
from model import TransformerVAE
from torch import nn
import numpy as np

def generate(model_path='./model_transformer_vae.pth', latent_dim=256, sequence_length=100, temperature=1.0):
    # Load the model
    model = TransformerVAE()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Generate a random latent vector
    z = torch.randn(1, latent_dim) * temperature

    # Prepare the latent vector for the decoder
    z = z.repeat(sequence_length, 1, 1)

    # Generate piano roll
    with torch.no_grad():
        generated_sequence, _, _ = model.decoder(z, z)  # Decoding from z to reconstruct the sequence

    # Convert the logits to binary values and transpose to match (num_pitches, time_steps)
    piano_roll = torch.sigmoid(generated_sequence).squeeze().t().numpy()
    piano_roll = np.where(piano_roll >= 0.5, 1, 0)

    return piano_roll

if __name__ == "__main__":
    piano_roll = generate()
    print("Generated Piano Roll Shape:", piano_roll.shape)
    # Optionally, save or visualize the piano roll here
