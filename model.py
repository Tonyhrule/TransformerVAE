import torch
from torch import nn

class TransformerVAE(nn.Module):
    def __init__(self, feature_size=128, latent_dim=128, num_heads=8, num_layers=4, dropout=0.1):
        super(TransformerVAE, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.to_latent = nn.Linear(feature_size, latent_dim * 2)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.to_output = nn.Linear(feature_size, feature_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, src):
        encoded = self.encoder(src)
        latent_params = self.to_latent(encoded.mean(dim=1))
        mu, logvar = latent_params.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        z = z.unsqueeze(1).expand(-1, src.size(1), -1)
        decoded = self.decoder(z, src)
        output = torch.sigmoid(self.to_output(decoded))  # Apply sigmoid to ensure output is between 0 and 1
        return output, mu, logvar
