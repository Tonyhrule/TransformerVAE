import torch
import numpy as np
import matplotlib.pyplot as plt
from model import TransformerVAE
from torch.utils.data import DataLoader
from postprocessing import npy_to_midi

# Function to prepare data for input into the Transformer model
def prepare_data(data, feature_size=128):
    if data.shape[1] < feature_size:
        padding = feature_size - data.shape[1]
        data = np.pad(data, ((0, 0), (0, padding)), 'constant')
    elif data.shape[1] > feature_size:
        data = data[:, :feature_size]
    return data

# Convert model output to MIDI-compatible values
def convert_to_midi_compatible(data):
    pitch = np.interp(data[:, 0], (data.min(), data.max()), (21, 108)).astype(int)
    start_time = np.cumsum(np.interp(data[:, 1], (data.min(), data.max()), (0, 1)))
    duration = np.interp(data[:, 2], (data.min(), data.max()), (0.1, 1))
    return np.vstack((pitch, start_time, duration)).T

# Function to generate MIDI from model output
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

    if generated_data.size > 0:
        midi_compatible_data = convert_to_midi_compatible(generated_data)
        np.save("temp_generated.npy", midi_compatible_data)
        npy_to_midi("temp_generated.npy", output_midi_path)
        print(f"Generated MIDI file saved to {output_midi_path}")
    else:
        print("No valid notes generated to convert into MIDI.")

# Function to analyze the variability of model outputs
def analyze_model_outputs(generated_data):
    plt.figure(figsize=(12, 6))
    plt.title('Model Output Analysis')
    plt.xlabel('Feature Index')
    plt.ylabel('Output Value')
    plt.boxplot(generated_data)
    plt.grid(True)
    plt.show()

# Main function to load the model and generate MIDI
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

    # Optional: Analyze the output for further tuning
    data = np.load(input_npy_path)
    prepared_data = prepare_data(data)
    data_tensor = torch.tensor(prepared_data, dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        generated, _, _ = model(data_tensor)
    analyze_model_outputs(generated.squeeze(0).cpu().numpy())

if __name__ == "__main__":
    main()
