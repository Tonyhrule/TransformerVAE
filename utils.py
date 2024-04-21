import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class PianoRollDataset(Dataset):
    def __init__(self, directory="./output", feature_size=128):
        self.file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
        self.feature_size = feature_size
        print(f"Loaded {len(self.file_paths)} datasets.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        if data.shape[1] != 3:
            raise ValueError(f"Expected data to have 3 features, but got {data.shape[1]}")

        # Transform the data from 3 features to exactly feature_size features
        expanded_data = np.tile(data, (1, self.feature_size // data.shape[1]))[:,:self.feature_size]
        return torch.tensor(expanded_data, dtype=torch.float32)

    @staticmethod
    def collate_fn(batch):
        batch = pad_sequence(batch, batch_first=True, padding_value=0)
        return batch

