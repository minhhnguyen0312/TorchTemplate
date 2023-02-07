import torch
import torch.nn.functional as F
import math
from utils import default
## Code for forward process

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

betas = cosine_beta_schedule(1000)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

def forwardProcess(x_start, timestep, noise=None):
    noise = default(noise, lambda: torch.randn_like(x_start))
    # Note the parameterization alpha = 1 - beta
    # X_t ~ N(x_t, sqrt(1 - alpha_t) * x_start, \bar{alpha}_t^2 * I)
    return (
        extract(alphas_cumprod, timestep, x_start.shape) * x_start +
        extract(alphas_cumprod_prev, timestep, x_start.shape) ** 2
    )