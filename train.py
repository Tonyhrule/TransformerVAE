import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os

from model import TransformerVAE
from utils import PianoRollDataset

class DummyScaler:
    def scale(self, loss):
        return loss
    def step(self, optimizer):
        optimizer.step()
    def update(self):
        pass

def train(model, data_loader, optimizer, epochs=10, device=torch.device('cpu'), save_path='', grad_accumulate_steps=1):
    scaler = GradScaler() if device.type == 'cuda' else DummyScaler()
    model.train()
    model.to(device)
    for epoch in range(epochs):
        total_loss = 0
        model.zero_grad()
        for batch_idx, data in enumerate(data_loader):
            data = data.to(device)
            with autocast(enabled=device.type == 'cuda'):
                recon_batch, mu, logvar = model(data)
                loss = model.loss_function(recon_batch, data, mu, logvar) / grad_accumulate_steps
            scaler.scale(loss).backward()
            if (batch_idx + 1) % grad_accumulate_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                model.zero_grad()
            total_loss += loss.item() * grad_accumulate_steps
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item()} * {grad_accumulate_steps}')
        print(f'Epoch {epoch+1}: Average Loss = {total_loss / len(data_loader)}')

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        torch.save(model.state_dict(), save_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerVAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
dataset = PianoRollDataset(directory="./output")
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=PianoRollDataset.collate_fn, num_workers=4, pin_memory=True)

save_path = r'C:\Users\tonyh\OneDrive\Desktop\MIDI-Music-Gen-AI\MIDI-Music-Gen-AI\ISEF-Music-AI-Ver.-MIDI-\TransformerVAE\model\transformer_vae_model.pth'
train(model, data_loader, optimizer, epochs=10, device=device, save_path=save_path, grad_accumulate_steps=4)
