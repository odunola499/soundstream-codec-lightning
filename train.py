import lightning as pl
import torch
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from omegaconf import OmegaConf
from soundstream import SoundStream
from losses import (
MultiFrequencyDiscriminator, GeneratorSTFTLoss, MSEDiscriminatorLoss
)
import random
from dataset import get_loaders
from datetime import datetime
from torch.optim.lr_scheduler import ExponentialLR
import argparse
from pytorch_lightning.loggers import WandbLogger

now = datetime.now()
millis_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")

project_name = "SoundStream"
experiment_name = f"{project_name}_{millis_string}"
train_config = OmegaConf.load('config.yaml')

def build_model(config = train_config):
    model = SoundStream(**config.generator.config)
    return model

def build_discriminator():
    discrim_config = train_config.mfd.config
    discriminator = MultiFrequencyDiscriminator(**discrim_config)
    return discriminator

class Module(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.generator = None
        self.discriminator = None
        self.loader = get_loaders(batch_size = 4)
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.bandwidths = train_config.generator.config.target_bandwidths
        self.generator_loss = None
        self.discriminator_loss = None
        self.build_criterion()


    def configure_model(self) -> None:
        self.generator = build_model()
        self.discriminator = build_discriminator()

    def build_criterion(self):
        d_config = train_config.criterion.g_criterion.config
        g_config = train_config.criterion.d_criterion.config
        self.generator_loss = GeneratorSTFTLoss(g_config)
        self.discriminator_loss = MSEDiscriminatorLoss()

    def configure_optimizers(self):
        g_config = train_config.optimizer.g.config
        d_config = train_config.optimizer.d.config
        g_sch_config = train_config.lr_scheduler.g.config
        d_sch_config = train_config.lr_scheduler.d.config

        g_optimizer = torch.optim.AdamW(self.generator.parameters(), **g_config)
        d_optimizer = torch.optim.AdamW(self.discriminator.parameters(), **d_config)

        g_scheduler = ExponentialLR(g_optimizer, **g_sch_config)
        d_scheduler = ExponentialLR(d_optimizer, **d_sch_config)

        return [g_optimizer, d_optimizer], [g_scheduler, d_scheduler]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        loader = self.loader
        return loader

    def training_step(self, batch, batch_idx):
        y = batch
        g_optimizer, d_optimizer = self.optimizers()
        g_scheduler, d_scheduler = self.lr_schedulers()
        bw = random.choice(self.bandwidths)

        self.toggle_optimizer(g_optimizer)
        g_optimizer.zero_grad()
        y_hat, commit_loss, last_layer = self.generator(y, bw = bw)
        output_real, output_fake, fmap_real, fmap_fake = self.discriminator(y, y_hat)
        g_loss, g_loss_items = self.generator_loss(y, y_hat,
                                                   output_real, output_fake,
                                                   fmap_real, fmap_fake,
                                                   use_adv_loss = True)
        g_loss = g_loss + (commit_loss * train_config.criterion.commit_loss_weight)
        g_loss_items['Train/commit_loss'] = commit_loss.item()
        g_loss_items['Train/g_loss'] = g_loss.item()
        g_loss.backward()
        g_optimizer.step()
        g_scheduler.step()
        self.untoggle_optimizer(g_optimizer)

        self.toggle_optimizer(d_optimizer)
        d_optimizer.zero_grad()
        d_loss_items = dict()
        y_hat, commit_loss, last_layer = self.generator(y, bw = bw)
        output_real, output_fake, fmap_real, fmap_fake = self.discriminator(y, y_hat)
        d_loss = self.discriminator_loss(output_real, output_fake)
        d_loss_items['Train/d_loss'] = d_loss.item()
        d_loss.backward()
        d_optimizer.step()
        d_scheduler.step()
        self.untoggle_optimizer(d_optimizer)

        self.log_dict(g_loss_items, prog_bar = True, on_step = True, enable_graph = True)
        self.log_dict(d_loss_items, prog_bar = True, on_step = True, enable_graph = True)
        self.log('train_loss', g_loss_items['Train/g_loss'])



logger = WandbLogger(project = project_name, name = experiment_name)
checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
    monitor='train_loss',
    dirpath='./checkpoints',
    mode="min",
    filename='{epoch:02d}-{train_loss:.2f}',
    every_n_train_steps=10000,
    save_weights_only=False)

learning_rate_monitor = pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step",
                                                                 log_momentum=True,
                                                                 log_weight_decay=True)

trainer = pl.Trainer(
    devices='auto',
    callbacks=[checkpoint_callback, learning_rate_monitor],
    enable_checkpointing=True,
    log_every_n_steps=5,
    num_nodes=1,
    accelerator="gpu",
    logger=logger,
    max_epochs=20,
    precision="bf16-mixed",
    accumulate_grad_batches=1)

module = Module()
trainer.fit(module)