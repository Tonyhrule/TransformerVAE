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
    data = prepare_data(data)
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)

    # Set model to evaluation mode and generate music
    model.eval()
    with torch.no_grad():
        generated, _, _ = model(data_tensor)

    # Check and print output data
    generated_data = generated.squeeze(0).cpu().numpy()
    print("Generated Data Shape:", generated_data.shape)
    print("Sample of Generated Data:", generated_data[:5])

    # Assume the output needs to be adjusted or scaled
    # Example: Convert to a proper MIDI format if necessary
    # This part depends heavily on your specific model output and needs
    if generated_data.shape[1] > 3:
        generated_data = generated_data[:, :3]

    generated_npy_path = "temp_generated.npy"
    np.save(generated_npy_path, generated_data)

    npy_to_midi(generated_npy_path, output_midi_path)
    print(f"Generated MIDI file saved to {output_midi_path}")

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = r'C:\Users\tonyh\OneDrive\Desktop\MIDI-Music-Gen-AI\MIDI-Music-Gen-AI\ISEF-Music-AI-Ver.-MIDI-\TransformerVAE\model\transformer_vae_model.pth'
    model = TransformerVAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Paths for the input and output files
    input_npy_path = 'output/midis-Alone.npy'
    output_midi_path = 'output/output_path_for_your_midi.mid'

    # Generate the MIDI file from the input NPY
    generate(input_npy_path, output_midi_path, model, device)

if __name__ == "__main__":
    main()
