import torch
import numpy as np
from model import TransformerVAE
from postprocessing import npy_to_midi

def prepare_data(data, feature_size=128):
    """
    Prepares data for input into the Transformer model.
    This function ensures the input data matches the required feature size by padding or truncating.
    """
    if data.shape[1] < feature_size:
        padding = feature_size - data.shape[1]
        data = np.pad(data, ((0, 0), (0, padding)), 'constant')
    elif data.shape[1] > feature_size:
        data = data[:, :feature_size]
    return data

def convert_to_midi_compatible(data, window_size=10, activation_scale=0.05, pitch_scale=88, base_note=21):
    """
    Convert the model output to MIDI-compatible values, using a sliding window to determine note activation.
    """
    midi_data = []
    time = 0
    for i in range(data.shape[0] - window_size):
        window = data[i:i+window_size, 1]  # Take a slice of activation data
        if np.mean(np.diff(window)) > activation_scale:  # Check if average change exceeds a scaled threshold
            pitch = int(np.interp(data[i, 0], [data[:, 0].min(), data[:, 0].max()], [base_note, base_note + pitch_scale]))
            duration = np.interp(data[i, 2], [data[:, 2].min(), data[:, 2].max()], [0.1, 1.5])
            midi_data.append([pitch, time, duration])
            time += duration

    return np.array(midi_data)

# Update the generate function to use this new conversion function
def generate(input_npy_path, output_midi_path, model, device):
    data = np.load(input_npy_path)
    data = prepare_data(data)
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        generated, _, _ = model(data_tensor)

    generated_data = generated.squeeze(0).cpu().numpy()
    print("Generated Data Shape:", generated_data.shape)
    print("Sample of Generated Data:", generated_data[:5])

    midi_compatible_data = convert_to_midi_compatible(generated_data)
    if midi_compatible_data.size > 0:
        generated_npy_path = "temp_generated.npy"
        np.save(generated_npy_path, midi_compatible_data)
        npy_to_midi(generated_npy_path, output_midi_path)
        print(f"Generated MIDI file saved to {output_midi_path}")
    else:
        print("No valid notes generated to convert into MIDI.")


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = r'C:\Users\tonyh\OneDrive\Desktop\MIDI-Music-Gen-AI\MIDI-Music-Gen-AI\ISEF-Music-AI-Ver.-MIDI-\TransformerVAE\model\transformer_vae_model.pth'
    model = TransformerVAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Paths for the input and output files
    input_npy_path = 'output\midis-Legend_of_Zelda_Links_Awakening_Overworld.npy'
    output_midi_path = 'generatedmusic.mid'

    # Generate the MIDI file from the input NPY
    generate(input_npy_path, output_midi_path, model, device)

if __name__ == "__main__":
    main()
