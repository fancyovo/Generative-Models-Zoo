import yaml
import os
from models import Generator, Discriminator
from train import GAN_Trainer
from dataset import get_mnist_dataloader
from utils.logger import setup_logging

if __name__ == '__main__':
    # Load config file
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    generator = Generator(latent_dim = config['latent_dim'],
                         img_shape = config['img_shape'],
                         channels = config['channels'])

    discriminator = Discriminator(img_shape = config['img_shape'],
                                 channels = config['channels'])

    dataloader = get_mnist_dataloader(config['batch_size'])
    trainer = GAN_Trainer(generator, discriminator, dataloader, config)
    trainer.train()