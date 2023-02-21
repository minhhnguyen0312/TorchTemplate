import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WSLinear(nn.Module):
    def __init__(
        self, in_features, out_features
    ):
        super(WSLinear,self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale  = (2/in_features) ** 0.5
        self.bias   = self.linear.bias
        self.linear.bias = None

        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self,x):
        return self.linear(x * self.scale) + self.bias

class WSConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1
    ):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (2 / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale   = WSLinear(w_dim, channels)
        self.style_bias    = WSLinear(w_dim, channels)

    def forward(self,x,w):
        x = self.instance_norm(x)
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias  = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return style_scale * x + style_bias

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super(NoiseInjection, self).__init__()
        # For generating batches of image (1, C, 1, 1)
        self.register_parameter("weight", torch.zeros((1, channels, 1, 1)))
    
    def forward(self, x):
        noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device = x.device)
        return x + self.weight * noise
    
class MLPStyle(nn.Module):
    def __init__(self, 
        latent_dim=512, 
        n_blocks=8,
    ):
        super(MLPStyle, self).__init__()
        self.model = nn.ModuleList(
           [nn.Linear(latent_dim, latent_dim) for i in range(n_blocks)]
        )

    def forward(self, z):
        for mod in self.model:
            z = nn.ReLU(mod(z))
        return self.model(z)

class GenBlock(nn.Module):
    def __init__(self, in_channel, out_channel, w_dim):
        super(GenBlock, self).__init__()
        self.conv1 = WSConv2d(in_channel, out_channel)
        self.conv2 = WSConv2d(out_channel, out_channel)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.inject_noise1 = NoiseInjection(out_channel)
        self.inject_noise2 = NoiseInjection(out_channel)
        self.adain1 = AdaIN(out_channel, w_dim)
        self.adain2 = AdaIN(out_channel, w_dim)
    def forward(self, x,w):
        x = self.adain1(self.leaky(self.inject_noise1(self.conv1(x))), w)
        x = self.adain2(self.leaky(self.inject_noise2(self.conv2(x))), w)
        return x

class StyleGenerator(nn.Module):
    def __init__(self, config, latent_dim, constant_dim, image_dim):
        super(StyleGenerator, self).__init__()
        self.config = config
        self.latent_dim = config['latent_dim']
        self.constant_dim = config['constant_dim']
        self.image_dim = config['image_dim']
        self.image_size = config['image_size']
        
        self.num_blocks = int(math.log2(self.image_size / 4))

        # Initialize Image
        self.constant_img = nn.Parameter(torch.ones((1, 3, self.constant_dim, 4)))
        self.adain1 = AdaIN(self, self.constant_dim, self.latent_dim)
        self.adain2 = AdaIN(self, self.constant_dim, self.latent_dim)
        self.initial_noise1 = NoiseInjection(self.constant_dim)
        self.initial_noise2 = NoiseInjection(self.constant_dim)
        self.initial_conv   = nn.Conv2d(self.constant_dim, self.constant_dim, kernel_size=3, stride=1, padding=1)
        self.leaky          = nn.LeakyReLU(0.2, inplace=True)

        # Body
        self.style = MLPStyle()
        self.gen = nn.ModuleList([
            GenBlock(self.latent_dim / 2 ** (i), self.latent_dim / 2 ** (i + 1), self.latent_dim) for i in range(config['num_block'])
        ])
        self.gen.add_module("rgb_layer", nn.Conv2d(self.latent_dim / 2 ** self.num_blocks, self.image_dim, kernel_size=1, stride=1, padding=0))
    
    def get_latent_code(self, n_samples):
        noise = torch.randn((self.latent_dim, n_samples))
        learned_latent = self.style(noise)
        return noise, learned_latent
    
    def sample(self, n_samples):
        z, w = self.get_noise(n_samples)

        return self(noise=z, latent=w)
    
    def forward(self, latent):
        w = latent
        x = self.constant_img
        x = self.initial_noise1(x)
        x = self.adain1(x, w)
        x = self.leaky(self.initial_conv(x))
        x = self.initial_noise2(x)
        out = self.adain2(x, w)

        for i, module in enumerate(self.gen):
            # Image size: 4 * 2**(i+1)
            # Channel: 
            upscaled = F.interpolate(out, scale_factor=2, mode='bilinear')
            out = module(upscaled)
        return out
        
