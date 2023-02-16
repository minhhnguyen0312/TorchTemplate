import torch
import torch.nn

from ..model import BaseModel

class BaseGAN(BaseModel):
    def __init__(self, config):
        super(BaseGAN, self).__init__(config)
        self.gen = self.build_generator()
        self.disc = self.build_disc()
    
    def build_generator(self):
        pass

    def build_discriminator(self):
        pass

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
        real = batch['image']
        b, w, h, c = real.shape
        real_label = torch.ones((b,))
        fake = self.sample(n_sample=b, cond=batch, grad=True)
        fake_label = torch.zeros((b,))
        
        real_loss = self.get_disc_loss(real, real_label)
        fake_loss = self.get_disc_loss(fake, fake_label)
        total_loss = (real_loss + fake_loss) / 2

        loss.


    def eval_step(self, batch, step):
        pass