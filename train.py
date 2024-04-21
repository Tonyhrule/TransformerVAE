# Import necessary modules
from utils import PianoRollDataset
from model import TransformerVAE
from torch.utils.data import DataLoader
from torch import optim

# Load the dataset
dataset = PianoRollDataset()
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=PianoRollDataset.collate_fn)

# Initialize the model
model = TransformerVAE(feature_size=128)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train the model
def train(model, data_loader, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in data_loader:
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = model.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}: Loss = {total_loss / len(data_loader)}')

train(model, data_loader, optimizer)
