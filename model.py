import torch
from torch import nn

class TransformerVAE(nn.Module):
    def __init__(self, feature_size=128, latent_dim=128, num_heads=8, num_layers=4, dropout=0.1):
        super(TransformerVAE, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.to_latent = nn.Linear(feature_size, latent_dim * 2)  # Outputs mu and logvar
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.to_output = nn.Linear(latent_dim, feature_size)  # Adjusted to ensure dimensions align

        # Initialize weights for better model performance
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, src):
        # Encoding
        encoded = self.encoder(src)
        latent_params = self.to_latent(encoded.mean(dim=1))
        mu, logvar = latent_params.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)

        # Decoding
        z = z.unsqueeze(1).repeat(1, src.size(1), 1)  # Repeat latent vector for each time step
        decoded = self.decoder(z, encoded)  # Use encoded as the memory state
        output = torch.sigmoid(self.to_output(decoded))  # Apply sigmoid to ensure output is between 0 and 1

        return output, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    