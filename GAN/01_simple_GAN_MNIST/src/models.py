import numpy as np
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape, channels):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.channels = channels

        self.dense = nn.Linear(self.latent_dim, 7*7*64)

        self.Sequential = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 7*7*64 -> 14*14*32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1), # 14*14*32 -> 28*28*16
            nn.Sigmoid()
        )

    def forward(self, z = None):
        if z is None:
            z = torch.randn(1, self.latent_dim)
        img = self.dense(z)
        img = img.reshape(img.shape[0], 64, 7, 7)
        img = self.Sequential(img)
        img = img.reshape(img.shape[0], *self.img_shape, self.channels)
        return img
    

class Discriminator(nn.Module):
    def __init__(self, img_shape, channels):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.channels = channels

        self.Sequential = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=3, stride=2, padding=1), # 28*28*1 -> 14*14*32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 14*14*32 -> 7*7*64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(7*7*64, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.reshape(img.shape[0], 1, 28, 28)
        validity = self.Sequential(img_flat)
        return validity

    

    