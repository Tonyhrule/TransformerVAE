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
        print(f'Epoch {epoch+1}: Average Loss = {total_loss / len(data_loader)}')

    # Ensure the directory exists, if not, create it
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Directory '{os.path.dirname(save_path)}' created or already exists.")
    except Exception as e:
        print(f"Failed to create or access the directory due to: {e}")
    
    # Attempt to save the model state
    try:
        torch.save(model.state_dict(), save_path)
        print(f'Model successfully saved to {save_path}')
    except Exception as e:
        print(f'Failed to save the model due to: {e}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerVAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
dataset = PianoRollDataset(directory="./output")
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=PianoRollDataset.collate_fn)

# Define the full path where the model will be saved
save_path = r'C:\Users\tonyh\OneDrive\Desktop\MIDI-Music-Gen-AI\MIDI-Music-Gen-AI\ISEF-Music-AI-Ver.-MIDI-\TransformerVAE\model\transformer_vae_model.pth'
train(model, data_loader, optimizer, device=device, save_path=save_path)

