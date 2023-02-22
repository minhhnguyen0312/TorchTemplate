import torch
import torch.nn as nn
from tqdm import tqdm
from random import random
from einops import reduce
from functools import partial
from collections import namedtuple

from diffusers import UNet2DConditionModel, DDPMScheduler
from .sampling import *
from ..model import BaseModel

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def identity(t, *args, **kwargs):
    return t

"""
Recall: DDPM contains 2 opposite Gaussian Markov chains
1. The appoximate posterior q(x_{1:T} | x_0) - diffusion process (forward process)
2. The reverse Gaussian Markov chains - denoising process (reverse process)
    p_{\theta}(x_{0:T}) = p(x_T) * \prod(p_{\theta}(x_{t-1} | x_t))
"""

class DDPM(BaseModel):
    def __init__(self, config, **kwargs):
        super(DDPM, self).__init__(config)
        self.model = UNet2DConditionModel()
        self.scheduler = DDPMScheduler(
                                    self.config['num_train_timesteps'],
                                    self.config['beta_start'],
                                    self.config['beta_end'],
                                    beta_schedule=self.config['beta_schedule'],
                                )
        self.register_schedules(max_steps=self.config['max_steps'])
        self.initialize()

    def register_schedules(self, max_steps=1000):
        betas = self.scheduler.betas
        alphas = self.scheduler.alphas
        alphas_cumprod = self.scheduler.alphas_cumprod

        p2_loss_weight_k = self.config.get('p2_loss_weight_k', 1)
        p2_loss_weight_gamma = self.config.get('p2_loss_weight_gamma', 0)

        self.register_buffer('betas', betas)
        self.register_buffer('alpha', alphas)
        self.register_buffer('alpha_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        self.register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)
    
    # Backward process
    def q_sample(self, x_start, t, noise=None):
        """Returns the mean of x_t's distribution"""
        noise = default(noise, torch.randn_like(x_start))
        return extract(
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # Forward
    def q_posterior(self, x_start, x_t, t):
        """Return q(x_{t-1} | x_t, x_0)"""
        posterior_mean = (
            extract(self.posterior_mean_coeff1, t, x_start.shape) * x_start +
            extract(self.posterior_mean_coeff2, t, x_start.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise=None):
        """ Predict x_0 from x_t, t
        returns:
            x_0_hat = x_t/sqrt(a_cump) - noise * sqrt(1/a_cump - 1)
        Note that x_t = sqrt(a_cump) * x_0 + noise / sqrt(1 - a_cump)
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def forward(self, img):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        # img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t)
    
    def p_losses(self, x_start, t, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()
        # predict and take gradient step
        model_out = self.model(x, t, x_self_cond) # e_theta
        target = noise

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def loss_fn(self):
        return nn.L1Loss()
    
    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False):
        """Forward to get e_theta, then compute p_theta(x_{t-1} | x_t)"""
        model_output = self.model(x, t, x_self_cond) # e_theta
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        pred_noise = model_output
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        x_start = maybe_clip(x_start)
        return ModelPrediction(pred_noise, x_start)
    
    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):
        """ Compute variance and mean for f_theta(x_{t-1} | x_t)
        Clamp image to [-1., 1.]
        Steps:
        1. model prediction to get mu_theta_t
        2. From q_posteriors variance => p_theta variance
        """
        preds = self.model_predictions(x, t, x_self_cond=x_self_cond) # Mean
        x_start = preds.pred_x_start
        if clip_denoised:
            x_start.clamp_(-1, 1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x,  t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
    
    @torch.no_grad()
    def p_sample(self, x, t, x_self_cond):
        """
        Sample x_{t-1} given x_t. 
        Where x_{t-1} ~ N(mu_theta, sigma)
        => sample x_{t-1} = mu_theta + sigma * noise
        """
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        noise = torch.randn_like(x) if t > 0 else 0.
        mean, _, log_variance, x_start = self.p_mean_variance(x, batched_times, x_self_cond=x_self_cond)

        image = mean + (0.5 * log_variance).exp() * noise
        return image, x_start
    
    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps=False):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        imgs = [img]
        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc="Sampling loop time step", total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)
        ret = img
        ret = self.unormalize(ret)
        return ret
    
    @torch.no_grad()
    def ddim_sample(self, shape):
        pass
    
    @torch.no_grad()
    def sample(self, batch_size=16, return_all_timesteps=False):
        image_size, channels = self.image_size, self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), return_all_timesteps=return_all_timesteps)
    
    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def train_step(self, batch, i):
        batch = self.cast_inputs(batch)
        loss = self(batch['images'])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    @torch.no_grad()
    def eval_step(self, batch, i):
        n_sample = batch['images'].shape[0]
        return self.sample(n_sample)
    
