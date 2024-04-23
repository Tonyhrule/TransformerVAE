import os

def test_path_write(save_path):
    # Ensure the directory exists, if not, create it
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Directory '{os.path.dirname(save_path)}' created or already exists.")
    except Exception as e:
        print(f"Failed to create or access the directory due to: {e}")
        return

    # Attempt to write a small test file
    try:
        with open(save_path, 'w') as f:
            f.write('Test write operation successful.')
        print(f'Test file successfully written to {save_path}')
    except Exception as e:
        print(f'Failed to write to the file due to: {e}')

# Define the full path for the test file
test_file_path = r'C:\Users\tonyh\OneDrive\Desktop\MIDI-Music-Gen-AI\MIDI-Music-Gen-AI\ISEF-Music-AI-Ver.-MIDI-\TransformerVAE\model\test_file.txt'
test_path_write(test_file_path)
