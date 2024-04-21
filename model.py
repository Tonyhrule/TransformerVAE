import torch
from torch import nn

class TransformerVAE(nn.Module):
    def __init__(self, feature_size=128, latent_dim=128, num_heads=8, num_layers=4, dropout=0.1):
        super(TransformerVAE, self).__init__()
        self.feature_size = feature_size
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.feature_size, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        # Change to_latent to output latent_dim * 2 but latent_dim should be same as feature_size
        self.to_latent = nn.Linear(self.feature_size, self.feature_size * 2)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.feature_size, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.to_output = nn.Linear(self.feature_size, self.feature_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, src):
        print("Input shape at start of forward:", src.shape)
        encoded = self.encoder(src)
        print("Encoded shape:", encoded.shape)
        latent_params = self.to_latent(encoded.mean(dim=1))
        mu, logvar = latent_params.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        z = z.unsqueeze(1).expand(-1, src.size(1), -1)
        print("Z shape after expand:", z.shape)
        decoded = self.decoder(z, src)
        print("Decoded shape:", decoded.shape)
        output = self.to_output(decoded)
        print("Output shape:", output.shape)
        return output, mu, logvar

def test_model():
    dummy_input = torch.randn(1, 10, 128)
    print("Dummy input shape:", dummy_input.shape)
    model = TransformerVAE(feature_size=128)
    try:
        output, mu, logvar = model(dummy_input)
        print("Input processed successfully. Output shape:", output.shape)
    except Exception as e:
        print("Error processing input:", str(e))

if __name__ == "__main__":
    test_model()
