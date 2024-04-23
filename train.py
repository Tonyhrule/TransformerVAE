import torch
from torch import optim
from torch.utils.data import DataLoader
import os

from model import TransformerVAE
from utils import PianoRollDataset

def train(model, data_loader, optimizer, epochs=10, device=torch.device('cpu'), save_path=''):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, data in enumerate(data_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = model.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item()}')

                # Direct save test after a batch
                try:
                    torch.save(model.state_dict(), save_path)
                    print(f'Test Model saved successfully after Batch {batch_idx}')
                except Exception as e:
                    print(f'Test Failed to save the model after Batch {batch_idx} due to: {e}')
                
        print(f'Epoch {epoch+1}: Average Loss = {total_loss / len(data_loader)}')

    # Check and create directory for saving model
    save_directory = os.path.dirname(save_path)
    try:
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        print(f"Directory '{save_directory}' created or already exists.")
    except Exception as e:
        print(f"Failed to create or access the directory due to: {e}")

    # Attempt to save the model state
    try:
        torch.save(model.state_dict(), save_path)
        print(f'Model successfully saved to {save_path}')
    except Exception as e:
        print(f'Failed to save the model at the end of training due to: {e}')

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the model instance
model = TransformerVAE()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Load the dataset and create a DataLoader instance
dataset = PianoRollDataset(directory="./output", max_files=100)  # Adjust '100' as needed
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=PianoRollDataset.collate_fn)

# Define the full path where the model will be saved
save_path = r'C:\Users\tonyh\OneDrive\Desktop\MIDI-Music-Gen-AI\MIDI-Music-Gen-AI\ISEF-Music-AI-Ver.-MIDI-\TransformerVAE\model\transformer_vae_model.pth'

# Run the training function
train(model, data_loader, optimizer, device=device, save_path=save_path)
