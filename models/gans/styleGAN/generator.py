import torch
import torch.nn as nn

class AdaIn(nn.Module):
    def __init__(self):
        super(AdaIn, self).__init__()
    
    def forward(self, x):
        pass

class MLPStyle(nn.Module):
    def __init__(self, 
        latent_dim=512, 
        n_blocks=8,
    ):
        super(MLPStyle, self).__init__()
        self.model = nn.Sequential(
            *[nn.Linear(latent_dim, latent_dim) for i in range(n_blocks)]
        )

    def forward(self, z):
        return self.model(z)

class StyleGenerator(nn.Module):
    def __init__(self, config):
        super(StyleGenerator, self).__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.style = MLPStyle()
    
    def get_noise(self, n_samples):
        noise = torch.randn((self.latent_dim, n_samples))
        learned_latent = self.style(noise)
        return noise, learned_latent
    
    def sample(self, n_samples):
        z, w = self.get_noise(n_samples)

        return self(noise=z, latent=w)
    
    def forward(self, noise, latent):
        pass
