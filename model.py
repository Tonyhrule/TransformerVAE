import torch
from torch import nn

class TransformerVAE(nn.Module):
    def __init__(self, feature_size=128, latent_dim=256, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.to_latent = nn.Linear(feature_size, latent_dim * 2)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.to_output = nn.Linear(latent_dim, feature_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, src):
        src = src.permute(1, 0, 2)  # Transformer expects (S, N, E)
        encoded = self.encoder(src)
        latent_params = self.to_latent(encoded.mean(dim=0))
        mu, logvar = latent_params.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        z = z.unsqueeze(0).repeat(encoded.size(0), 1, 1)
        decoded = self.decoder(z, src)
        return decoded.permute(1, 0, 2), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        from torch.nn import functional as F
        BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
