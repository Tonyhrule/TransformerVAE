import numpy as np
import os

def check_and_fix_npy_files(directory, expected_features=128):
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            filepath = os.path.join(directory, filename)
            data = np.load(filepath)
            if data.shape[1] != expected_features:
                print(f"Fixing file {filename}: expected {expected_features} features, got {data.shape[1]}")
                # Assuming the transpose fixes the issue; otherwise, adapt accordingly.
                if data.shape[0] == expected_features:
                    data = data.T
                np.save(filepath, data)

directory = "./output"  # Change this to your directory containing .npy files
check_and_fix_npy_files(directory)
