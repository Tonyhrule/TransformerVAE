import torch
import numpy as np
import os
from model import TransformerVAE
from postprocessing import npy_to_midi

def prepare_data(data, feature_size=128):
    """
    Prepares data for input into the Transformer model.
    This function ensures the input data matches the required feature size by padding or truncating.
    """
    if data.shape[1] < feature_size:
        # Pad the data if less than required feature size
        padding = feature_size - data.shape[1]
        data = np.pad(data, ((0, 0), (0, padding)), 'constant')
    elif data.shape[1] > feature_size:
        # Truncate the data if more than required feature size
        data = data[:, :feature_size]
    return data

def generate(input_npy_path, output_midi_path, model, device=torch.device('cpu')):
    """
    Generate MIDI file from NPY input using a trained model.
    :param input_npy_path: Path to the input NPY file.
    :param output_midi_path: Path to save the output MIDI file.
    :param model: Trained model instance.
    :param device: Computation device.
    """
    # Load data
    data = np.load(input_npy_path)
    data = prepare_data(data)  # Prepare the data to ensure it has the correct feature size
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)

    # Set model to evaluation mode and generate music
    model.eval()
    with torch.no_grad():
        generated, _, _ = model(data_tensor)

    # Assuming the output is in the correct format [batch_size, seq_length, feature_dim]
    # and feature_dim should be at least 3 (for pitch, start_tick, duration_tick)
    generated_data = generated.squeeze(0).cpu().numpy()

    # Here we need to ensure that only the necessary columns (pitch, start_tick, duration_tick) are extracted
    if generated_data.shape[1] > 3:
        generated_data = generated_data[:, :3]  # Assuming the first three columns are the needed ones

    # Convert the tensor back to numpy and save as npy
    generated_npy_path = "temp_generated.npy"
    np.save(generated_npy_path, generated_data)

    # Convert the NPY data to MIDI
    npy_to_midi(generated_npy_path, output_midi_path)
    print(f"Generated MIDI file saved to {output_midi_path}")

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerVAE().to(device)

    # Paths for the input and output files
    input_npy_path = 'output\midis-Alone.npy'
    output_midi_path = 'output_path_for_your_midi.mid'

    # Generate the MIDI file from the input NPY
    generate(input_npy_path, output_midi_path, model, device)

if __name__ == "__main__":
    main()
