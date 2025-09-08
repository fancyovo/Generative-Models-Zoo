import numpy as np
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape, channels):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.channels = channels

        self.dense = nn.Linear(self.latent_dim, 1024*16*16)

        self.Sequential = nn.Sequential(
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1), # 16*16*1024 -> 32*32*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 32*32*512 -> 64*64*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 64*64*256 -> 128*128*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 128*128*128 -> 256*256*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, self.channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z = None):
        if z is None:
            z = torch.randn(1, self.latent_dim)
        img = self.dense(z)
        img = img.reshape(img.shape[0], 1024, 16, 16)
        img = self.Sequential(img)
        img = img.reshape(img.shape[0], self.channels, *self.img_shape)
        return img
    

class Discriminator(nn.Module):
    def __init__(self, img_shape, channels):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.channels = channels

        self.Sequential = nn.Sequential(
            nn.Conv2d(self.channels, 128, kernel_size=4, stride=2, padding=1), # 256*256*3 -> 128*128*128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1), # 128*128*128 -> 64*64*128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 64*64*128 -> 32*32*256
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1), # 32*32*256 -> 16*16*256
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # 16*16*256 -> 8*8*512
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1), # 8*8*512 -> 4*4*1024
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(4*4*1024, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.reshape(img.shape[0], 3, 256, 256)
        validity = self.Sequential(img_flat)
        return validity

    

    