import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from utils.logger import setup_logging
from torchvision.utils import save_image
import os

class GAN_Trainer:
    def __init__(self, generator, discriminator, data_loader, config):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.data_loader = data_loader
        self.config = config
        self.device = device

        log_dir = os.path.join(config['output_dir'], 'logs')
        self.logger = setup_logging(log_dir, f'{config["project_name"]}.log')

        self.logger.info(f'Initialized Trainer with config: {config}')
        self.logger.info(f'Using device: {device}')

    def train(self):
        self.logger.info(f'Starting training for {self.config["num_epochs"]} epochs')
        optimizer_G = optim.Adam(self.generator.parameters(), lr=self.config['lr'], 
                                 betas=(self.config['beta1'], self.config['beta2']))
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.config['lr'], 
                                 betas=(self.config['beta1'], self.config['beta2']))

        for epoch in range(self.config['num_epochs']):
            for i, (real_images, _) in enumerate(self.data_loader):
                # Train discriminator
                self.discriminator.zero_grad()
                real_images = real_images.to(self.device)
                batch_size = real_images.size(0)

                # Generate fake images
                z = torch.randn(batch_size, self.config['latent_dim']).to(self.device)
                fake_images = self.generator(z)

                # Discriminator loss
                real_validity = self.discriminator(real_images)
                fake_validity = self.discriminator(fake_images)
                d_loss = - (torch.mean(torch.log(real_validity) + torch.log(1 - fake_validity)))

                d_loss.backward()
                optimizer_D.step()

                # Train generator
                self.generator.zero_grad()
                z = torch.randn(batch_size, self.config['latent_dim']).to(self.device)
                fake_images = self.generator(z)
                fake_validity = self.discriminator(fake_images)
                g_loss = torch.mean(torch.log(1 - fake_validity))

                g_loss.backward()
                optimizer_G.step()
                
                if (i+1)%self.config['log_interval']==0:
                    self.logger.info(f'Epoch [{epoch+1}/{self.config["num_epochs"]}], Step [{i+1}/{len(self.data_loader)}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

            # save images
            z = torch.randn(100, self.config['latent_dim']).to(self.device)
            fake_images = self.generator(z)
            print(f'fake_images.shape: {fake_images.shape}')
            fake_images = fake_images.permute(0, 3, 1, 2)
            print(f'permuted fake_images.shape: {fake_images.shape}')
            if not os.path.exists(f'{self.config["output_dir"]}/fake_images'):
                os.makedirs(f'{self.config["output_dir"]}/fake_images')
            save_image(fake_images.data[:25], f'{self.config["output_dir"]}/fake_images/fake_images_{epoch+1}.png', nrow=5, normalize=True)
            self.logger.info(f'Saved fake images for epoch {epoch+1}, file:{self.config["output_dir"]}/fake_images/fake_image_{epoch+1}.png')
        

            # save models
            torch.save(self.generator.state_dict(), f'{self.config["output_dir"]}/generator.pth')
            torch.save(self.discriminator.state_dict(), f'{self.config["output_dir"]}/discriminator.pth')
            self.logger.info(f'Saved models for epoch {epoch+1}, files: {self.config["output_dir"]}/generator.pth, {self.config["output_dir"]}/discriminator.pth')

        self.logger.info(f'Training complete.')

            



                    
