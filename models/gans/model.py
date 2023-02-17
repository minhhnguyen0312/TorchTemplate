import torch
import torch.nn as nn

from ..model import BaseModel

class SimpleGenerator(nn.Module):
    def __init__(self):
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
    def __init__(self):
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
    def __init__(self, config):
        super(BaseGAN, self).__init__(config)
        self.gen, self.gen_optimizer = self.build_generator()
        self.disc, self.disc_optimizer = self.build_discriminator()
    
    def build_generator(self):
        gen = SimpleGenerator()
        optimizer = self.opt_mod(gen.parameters(), lr=0.001)
        return gen, optimizer

    def build_discriminator(self):
        disc = Discriminator()
        optimizer = self.opt_mod(disc.parameters(), lr=0.001)
        return disc, optimizer

    def train_step(self, batch, step):
        disc_loss = self.train_disc_step(batch)
        gen_loss = self.train_gen_step(batch)
        return {
            "disc_loss": disc_loss,
            "gen_loss": gen_loss
        }
    
    def train_disc_step(self, batch):
        self.gen.eval()
        self.disc.train()
        self.disc.optimizer.zero_grad()
        real = batch['image']
        b, w, h, c = real.shape
        real_label = torch.ones((b,))
        fake = self.sample(n_sample=b, cond=batch, grad=True)
        fake_label = torch.zeros((b,))
        
        real_loss = self.get_disc_loss(real, real_label)
        fake_loss = self.get_disc_loss(fake, fake_label)
        total_loss = (real_loss + fake_loss) / 2

        total_loss.backward()
        self.disc_optimizer.step()
    
    def train_gen_step(self, batch):
        self.gen.train()
        self.disc.eval()
        self.gen_optimizer.zero_grad()
        real = batch['image']
        b, w, h, c = real.shape
        fake = self.sample(n_sample=b, cond=batch, grad=True)
        fake_real_label = torch.ones((b,))
        
        gen_loss = self.get_disc_loss(fake, fake_real_label)
        gen_loss.backward()
        self.gen.optimizer.step()

    def sample(self, n_sample, cond=None, grad=True):
        noise = torch.randn(n_sample, 100, device=self.device)
        fake = self.gen(noise)
        return fake
    
    def get_disc_loss(self, x, y):
        pred = self.disc(x)
        return self.criterion(pred, y)
    
    def eval_step(self, batch, step):
        pass
