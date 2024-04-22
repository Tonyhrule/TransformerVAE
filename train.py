import torch
from torch import optim
from torch.utils.data import DataLoader
import os

from model import TransformerVAE
from utils import PianoRollDataset

def train(model, data_loader, optimizer, epochs=10, device=torch.device('cpu'), save_path='/model/transformer_vae_model.pth'):
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

    # Check if the directory exists, if not, create it
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the model state
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

train(model, data_loader, optimizer, device=device, save_path='/model/transformer_vae_model.pth')
