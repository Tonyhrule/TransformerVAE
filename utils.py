import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class PianoRollDataset(Dataset):
    """
    Loads and processes piano roll data from NPY files into tensors for model input, adjusting each sample to a specified feature size.

    Attributes:
        directory (str): Directory containing NPY files.
        feature_size (int): Target size for each sample's features.
        max_files (int): Limit on the number of files to load.
    """

    def __init__(self, directory="./output", feature_size=128, max_files=300):
        super(PianoRollDataset, self).__init__()
        self.directory = directory
        self.feature_size = feature_size
        self.max_files = max_files
        self.file_paths = self.load_file_paths()

    def load_file_paths(self):
        """Loads up to max_files NPY file paths from the directory."""
        try:
            files = [os.path.join(self.directory, f) for f in os.listdir(self.directory) if f.endswith('.npy')]
            return files[:self.max_files]
        except Exception as e:
            raise RuntimeError(f"Unable to access {self.directory}: {e}")

    def __len__(self):
        """Returns the number of loaded files."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Retrieves and processes the piano roll data from a file.

        Args:
            idx (int): Index of the file in the loaded list.

        Returns:
            torch.Tensor: Processed data tensor for model input.
        """
        try:
            data = np.load(self.file_paths[idx])
            if data.shape[1] != self.feature_size:
                data = np.pad(data, ((0, 0), (0, self.feature_size - data.shape[1])), mode='constant', constant_values=0)
            return torch.tensor(data / max(1, np.max(data)), dtype=torch.float32)
        except Exception as e:
            raise RuntimeError(f"Error processing file {self.file_paths[idx]}: {e}")

    @staticmethod
    def collate_fn(batch):
        """Combines data items into a batch, padding as needed to match the largest item's size."""
        return pad_sequence(batch, batch_first=True, padding_value=0)
