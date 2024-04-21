import os
from preprocessing import process_midi  # Corrected import statement
import torch

class MIDIDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, fs=100, seq_length=500):
        self.data = []
        for path in file_paths:
            piano_roll = process_midi(path, fs=fs)  # Corrected function call
            if piano_roll.shape[1] >= seq_length:  # Ensure piano roll is long enough
                self.data.append(piano_roll[:, :seq_length])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

def load_data(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mid"):
                file_list.append(os.path.join(root, file))
    return file_list
