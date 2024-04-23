import torch
from torch import nn

class TransformerVAE(nn.Module):
    """
    Transformer Variational Autoencoder (VAE) designed for sequence modeling and generation.
    
    This model combines the architecture of Transformers with the generative capabilities
    of a variational autoencoder, making it suitable for tasks such as time-series prediction,
    music generation, and other sequence generation tasks.

    Parameters:
        feature_size (int): The dimensionality of input features.
        latent_dim (int): The size of the latent space.
        num_heads (int): The number of attention heads in each Transformer layer.
        num_layers (int): The number of Transformer layers in the encoder and decoder.
        dropout (float): Dropout rate for regularization during training.
    """

    def __init__(self, feature_size=128, latent_dim=128, num_heads=8, num_layers=4, dropout=0.1):
        super(TransformerVAE, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.to_latent = nn.Linear(feature_size, latent_dim * 2)  # Outputs mu and log variance
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.to_output = nn.Linear(latent_dim, feature_size)

        self.initialize_weights()

    def initialize_weights(self):
        """
        Initializes weights with the Xavier uniform method and biases to zero,
        which is often a good practice for maintaining stability during training.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def reparameterize(self, mu, logvar):
        """
        Performs the reparameterization trick by sampling from a standard normal
        and scaling by standard deviation and shifting by the mean.
        
        Args:
            mu (Tensor): The mean vector of the latent Gaussian.
            logvar (Tensor): The logarithm of the variance vector of the latent Gaussian.

        Returns:
            Tensor: The sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, src):
        """
        Defines the forward pass of the VAE.

        Args:
            src (Tensor): The input tensor of shape (batch_size, sequence_length, feature_size).

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Reconstructed output, mu, and logvar tensors.
        """
        encoded = self.encoder(src)
        latent_params = self.to_latent(encoded.mean(dim=1))
        mu, logvar = latent_params.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        
        z = z.unsqueeze(1).repeat(1, src.size(1), 1)
        decoded = self.decoder(z, encoded)
        output = torch.sigmoid(self.to_output(decoded))
        return output, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        """
        Calculates the loss function which includes the reconstruction loss and
        the KL divergence loss for the VAE.

        Args:
            recon_x (Tensor): Reconstructed outputs.
            x (Tensor): Original inputs.
            mu (Tensor): Mean from the latent space.
            logvar (Tensor): Log variance from the latent space.

        Returns:
            Tensor: The total loss for the batch.
        """
        BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
