import torch
from torch import optim
from torch.utils.data import DataLoader

from model import TransformerVAE
from utils import PianoRollDataset


def train(model, data_loader, optimizer, epochs=1, device=torch.device('cpu')):
    model.train()
    model.to(device)
    batch_limit = 5  # Limit to a few batches for testing
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, data in enumerate(data_loader):
            if batch_idx >= batch_limit:
                break  # Stop after a few batches
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = model.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f'Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item()}')
        print(f'Epoch {epoch+1}: Average Loss = {total_loss / batch_limit}')

# Parameters and initialization for testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerVAE()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
dataset = PianoRollDataset(directory="./output")
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=PianoRollDataset.collate_fn)

# Run the small test
train(model, data_loader, optimizer, device=device)
