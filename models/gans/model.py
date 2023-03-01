import torch
import torch.nn as nn

from ..model import BaseModel

class SimpleGenerator(nn.Module):
    def __init__(self, config):
        super(SimpleGenerator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 784)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = x.view(-1, 1, 28, 28)
        return x


# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class BaseGAN(BaseModel):
    def __init__(self, config, **kwargs):
        super(BaseGAN, self).__init__(config, **kwargs)
        self.initialize(**kwargs)
    
    def initialize(self, **kwargs):
        # self.device = 'cpu'
        self.criterion = nn.BCELoss()
        gen_config = self.config['generator']
        disc_config = self.config['discriminator']
        self.gen, self.gen_optimizer = self.build_generator(gen_config)
        self.disc, self.disc_optimizer = self.build_discriminator(disc_config)

    def build_generator(self, config):
        opt_config = config['optimizer']
        gen_opt_mod = torch.optim.Adam
        gen = SimpleGenerator(config)
        gen.to(self.device)
        optimizer = gen_opt_mod(gen.parameters(), lr=opt_config['lr'])
        return gen, optimizer

    def build_discriminator(self, config):
        opt_config = config['optimizer']
        disc_opt_mod = torch.optim.Adam
        disc = Discriminator(config)
        disc.to(self.device)
        optimizer = disc_opt_mod(disc.parameters(), lr=opt_config['lr'])
        return disc, optimizer

    def train_step(self, batch, step):
        batch = self.cast_inputs(batch)
        disc_loss = self.train_disc_step(batch)
        gen_loss = self.train_gen_step(batch)
        return {
            "disc_loss": disc_loss,
            "gen_loss": gen_loss
        }
    
    def train_disc_step(self, batch):
        # print(batch[0].shape)
        self.gen.eval()
        self.disc.train()
        self.disc_optimizer.zero_grad()
        real = batch['image']
        b, w, h, c = real.shape
        real_label = torch.ones((b, 1), device=self.device)
        fake = self.sample(n_sample=b, cond=batch, grad=True)
        fake_label = torch.zeros((b, 1), device=self.device)
        
        # real_loss = self.get_disc_loss(real, real_label)
        fake_loss = self.get_disc_loss(fake, fake_label, fake=True)
        real_loss = self.get_disc_loss(real, real_label, fake=False)
        total_loss = (real_loss + fake_loss) / 2

        total_loss.backward()
        self.disc_optimizer.step()
        return total_loss
    
    def train_gen_step(self, batch):
        # print(batch)
        self.gen.train()
        self.disc.eval()
        self.gen_optimizer.zero_grad()
        real = batch['image']
        b, w, h, c = real.shape
        fake = self.sample(n_sample=b, cond=batch, grad=True)
        fake_real_label = torch.ones((b, 1), device=self.device)
        
        gen_loss = self.get_disc_loss(fake, fake_real_label, fake=True)
        gen_loss.backward()
        self.gen_optimizer.step()
        return gen_loss
    
    def sample(self, n_sample, cond=None, grad=False):
        noise = torch.randn(n_sample, self.config['generator']['latent_dim'], device=self.device)
        if not grad:
            with torch.no_grad():
                fake = self.gen(noise, depth=0, alpha=0)
        else:
            fake = self.gen(noise, depth=0, alpha=0)
        return fake

    
    def get_disc_loss(self, x, y, **kwargs):
        pred = self.disc(x, **kwargs)
        return self.criterion(pred, y)
    
    def eval_one_epoch(self, **kwargs):
        print("Sample are generated inside training function")
        return
