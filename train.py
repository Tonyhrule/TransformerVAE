import torch
from torch import optim
from torch.utils.data import DataLoader
from model import TransformerVAE
from utils import MIDIDataset, load_data

def train(model, data_loader, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch in data_loader:
            batch = batch.permute(0, 2, 1)  # Adjust shape to fit model input (N, E, S)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = model.loss_function(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch}, Loss {loss.item()}')

if __name__ == "__main__":
    file_paths = load_data('./midis')
    dataset = MIDIDataset(file_paths)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = TransformerVAE()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train(model, data_loader, optimizer)
