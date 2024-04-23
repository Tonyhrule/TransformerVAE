import torch
import numpy as np
from model import TransformerVAE
from postprocessing import npy_to_midi
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_data(data, feature_size=128):
    """
    Prepares data for input into the Transformer model by ensuring the input data
    matches the required feature size by padding or truncating.
    
    Args:
        data (np.array): Original data array.
        feature_size (int): Required feature size for the model input.
    
    Returns:
        np.array: Adjusted data array with the specified feature size.
    """
    if data.shape[1] < feature_size:
        padding = feature_size - data.shape[1]
        data = np.pad(data, ((0, 0), (0, padding)), 'constant')
    elif data.shape[1] > feature_size:
        data = data[:, :feature_size]
    return data

def convert_to_midi_compatible(data, window_size=10, activation_scale=0.01, pitch_scale=88, base_note=21):
    """
    Converts the model output to MIDI-compatible values using a sliding window
    to determine note activation.
    
    Args:
        data (np.array): The data to convert.
        window_size (int): The size of the sliding window for activation checks.
        activation_scale (float): Threshold for determining if a note is active.
        pitch_scale (int): Scale for mapping the model output to MIDI pitches.
        base_note (int): Base MIDI note for pitch calculation.
    
    Returns:
        np.array: MIDI data ready for conversion to an actual MIDI file.
    """
    midi_data = []
    time = 0
    for i in range(data.shape[0] - window_size):
        window = data[i:i+window_size, 1]
        if np.mean(np.diff(window)) > activation_scale:
            pitch = int(np.interp(data[i, 0], [data[:, 0].min(), data[:, 0].max()], [base_note, base_note + pitch_scale]))
            duration = np.interp(data[i, 2], [data[:, 2].min(), data[:, 2].max()], [0.1, 1.5])
            midi_data.append([pitch, time, duration])
            time += duration

    return np.array(midi_data)

def generate(input_npy_path, output_midi_path, model, device):
    """
    Generates MIDI from an input NPY file using a trained Transformer VAE model.

    Args:
        input_npy_path (str): Path to the input NPY file.
        output_midi_path (str): Path where the output MIDI file will be saved.
        model (TransformerVAE): Trained Transformer VAE model.
        device (torch.device): Device to perform computations on.
    """
    try:
        data = np.load(input_npy_path)
        data = prepare_data(data)
        data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            generated, _, _ = model(data_tensor)

        generated_data = generated.squeeze(0).cpu().numpy()
        logging.info(f"Generated Data Shape: {generated_data.shape}")
        logging.debug(f"Sample of Generated Data: {generated_data[:5]}")

        midi_compatible_data = convert_to_midi_compatible(generated_data)
        if midi_compatible_data.size > 0:
            generated_npy_path = "temp_generated.npy"
            np.save(generated_npy_path, midi_compatible_data)
            npy_to_midi(generated_npy_path, output_midi_path)
            logging.info(f"Generated MIDI file saved to {output_midi_path}")
        else:
            logging.info("No valid notes generated to convert into MIDI.")
    except Exception as e:
        logging.error(f"Failed to generate MIDI due to an error: {e}")

def main():
    """
    Main function to set up the environment, load the model, and generate MIDI.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = './model/transformer_vae_model.pth'
    model = TransformerVAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    input_npy_path = 'output/midis-Legend_of_Zelda_Links_Awakening_Overworld.npy'
    output_midi_path = 'generatedmusic.mid'
    generate(input_npy_path, output_midi_path, model, device)

if __name__ == "__main__":
    main()
