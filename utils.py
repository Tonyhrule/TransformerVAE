import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class PianoRollDataset(Dataset):
    def __init__(self, directory="./output", feature_size=128, max_files= 300):
        file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
        if max_files is not None:
            file_paths = file_paths[:max_files]
        self.file_paths = file_paths
        self.feature_size = feature_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        if data.shape[1] != self.feature_size:
            data = np.pad(data, ((0, 0), (0, self.feature_size - data.shape[1])), mode='constant', constant_values=0)
        # Normalizing data
        max_val = np.max(data)
        min_val = np.min(data)
        data = data / max_val if max_val > 0 else data
        return torch.tensor(data, dtype=torch.float32)

    @staticmethod
    def collate_fn(batch):
        return pad_sequence(batch, batch_first=True, padding_value=0)

