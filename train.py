from model import TransformerVAE
from utils import PianoRollDataset
from torch.utils.data import DataLoader
from torch import optim
import torch

def loss_function(recon_x, x, mu, logvar):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(model, data_loader, optimizer, epochs=10, device='cpu'):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        total_loss = 0
        for data in data_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}: Loss = {total_loss / len(data_loader)}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerVAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
dataset = PianoRollDataset(directory="./output")
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=PianoRollDataset.collate_fn)

train(model, data_loader, optimizer, device=device)
