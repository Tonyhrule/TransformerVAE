import torch
from torch import optim
from torch.utils.data import DataLoader
import os
import logging
from model import TransformerVAE
from utils import PianoRollDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(model, data_loader, optimizer, epochs=10, device=torch.device('cpu'), save_path='', validate=False, log_interval=100):
    """
    Train the TransformerVAE model with the given data.
    Optionally perform validation if validate set to True.

    Args:
        model (nn.Module): The TransformerVAE instance.
        data_loader (DataLoader): DataLoader providing the dataset.
        optimizer (Optimizer): Optimizer for training.
        epochs (int): Number of epochs to train.
        device (torch.device): Device to run the training on.
        save_path (str): Path where the model will be saved.
        validate (bool): Whether to perform validation using a validation set.
    """
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

            if batch_idx % log_interval == 0:
                logging.info(f'Epoch {epoch + 1}, Batch {batch_idx}: Loss = {loss.item()}')
                if batch_idx % (10 * log_interval) == 0:
                    save_model(model, save_path, batch_idx)

        average_loss = total_loss / len(data_loader)
        logging.info(f'Epoch {epoch + 1}: Average Loss = {average_loss}')

        if validate:
            validate_model(model, data_loader, device)

def save_model(model, save_path, batch_idx):
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logging.info(f'Model saved successfully after Batch {batch_idx} to {save_path}')
    except Exception as e:
        logging.error(f'Failed to save the model after Batch {batch_idx} due to: {e}')

def validate_model(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        validation_loss = sum(model.loss_function(*model(data.to(device))) for data in data_loader) / len(data_loader)
    logging.info(f'Validation Loss: {validation_loss}')
    model.train()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerVAE()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset = PianoRollDataset(directory="./output", max_files=100)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=PianoRollDataset.collate_fn)
    save_path = r'C:\Users\tonyh\OneDrive\Desktop\TransformerVAE\model\transformer_vae_model.pth'
    train(model, data_loader, optimizer, device=device, save_path=save_path, validate=True)
