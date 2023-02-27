import os
import torch
from utils.args import PythonParser, YmlParser
from models import build_model_from_config
from data import build_data_from_config
from tqdm import tqdm

class BaseTrainer:
    def __init__(self):
        self.model = None
        self.trn_data = None
        self.val_data = None
        self.config = "None"
        self.initialize()
    
    def initialize(self):
        raise NotImplementedError

    def fit(self, num_epochs=10, train_steps=100000, train_on_steps=False):
        if not train_on_steps:
            for i in range(num_epochs):
                self.model.train_one_epoch(self.trn_data, i)
                self.model.eval_one_epoch(self.val_data, i)
        else:
            steps = 0
            with tqdm(initial=steps, total=train_steps) as pbar:
                while steps < train_steps:
                    total_loss = 0.
                    data = next(self.trn_data)
                    loss = self.model.train_step(data, steps)
                    total_loss += loss.item()

                    pbar.set_description(f'loss: {total_loss:.4f}')
                    steps += 1

                    if steps != 0 and steps % self.config['log_step'] == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = steps // self.config['log_step']
                            all_images_list = list(map(lambda n: self.model.sample(batch_size=n), 16))

                        all_images = torch.cat(all_images_list, dim = 0)
                        # utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                        self.save(milestone)

                    pbar.update(1)
    def save(self, milestone):
        self.model.save()


class TrainerYml(BaseTrainer):
    def __init__(self, taskfile):
        self.taskfile = taskfile
        self.parser = YmlParser()
        super(TrainerYml, self).__init__()
    
    def initialize(self):
        self.config = self.parser.read(self.taskfile)
        self.initialize_model()
        self.initialize_data()
    
    def initialize_model(self):
        self.model = build_model_from_config(self.config['model'], device=self.config['task']['device'])
        self.model.to(self.config['task']['device'])

    def initialize_data(self):
        self.trn_data, self.val_data = build_data_from_config(self.config['data'])


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        convert_image_to = None
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader

        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    
                    # Train step
                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim = 0)
                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')


class DiffusionTrainerv2:
    def __init__(self):
        self.accelerator = None # Optional for mixed precision
    
        self.model = None
        self.train_data = None
