import numpy as np
import pandas as pd
import PIL
from PIL import Image
import cv2
import os, os.path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(100, 64*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64*2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def feed_forward(self, inp):
        x = self.seq(inp)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.seq = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*4) x 8 x 8
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*8) x 4 x 4
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def feed_forward(self, x):
        x = self.seq(x)
        return x

class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
        self.generator = Generator().float().cuda()
        self.discriminator = Discriminator().float().cuda()
        self.generator_optim = optim.Adam(self.generator.parameters())
        self.discriminator_optim = optim.Adam(self.discriminator.parameters())
        
    def generate_images(self, batch_size):
        inp = Variable(torch.randn(batch_size,100,1,1))
        out = self.generator.feed_forward(inp)
        return out
    
    def train(self, images, epochs = 20):
        print('============ Starting training of GAN ============')
        batch_size = 10
        loss = nn.BCELoss()
        for epoch in range(epochs):
            discriminator_error = 0
            generator_error = 0
            for i in range(int(len(images)/batch_size) + 1):
                self.generator_optim.zero_grad()
                self.discriminator_optim.zero_grad()
                try:
                    orig_images = torch.from_numpy(images[i*batch_size:(i+1)*batch_size, :, :, :]).permute(0, 3, 1, 2).float().cuda()
                    if(orig_images.shape[0]==0):
                        break
                except:
                    orig_images = torch.from_numpy(images[i*batch_size:]).permute(0, 3, 1, 2).float().cuda()
                
                #Training the discriminator
                fake_images_truth = Variable(torch.zeros(batch_size)).float().cuda()
                noise = Variable(torch.randn(batch_size, 100, 1, 1)).float().cuda()
                gen_images = self.generator.feed_forward(noise).cuda()
                fake_images_prediction = self.discriminator.feed_forward(gen_images)
                dis_error_fake = loss(fake_images_prediction, fake_images_truth)
                dis_error_fake.backward()
                
                true_images_truth = Variable(torch.ones(len(orig_images))).float().cuda()
                true_images_prediction = self.discriminator.feed_forward(orig_images)
                dis_error_true = loss(true_images_prediction, true_images_truth)
                dis_error_true.backward()
                self.discriminator_optim.step()
                
                discriminator_error = discriminator_error + dis_error_fake + dis_error_true
                
                #Training the generator
                
                noise = Variable(torch.randn(batch_size, 100, 1, 1)).float().cuda()
                gen_images = self.generator.feed_forward(noise)
                fake_images_prediction = self.discriminator.feed_forward(gen_images.cuda())
                fake_images_truth = Variable(torch.ones(batch_size)).float().cuda()
                dis_error_fake = loss(fake_images_prediction, fake_images_truth)
                dis_error_fake.backward()
                self.generator_optim.step()
                gen_error = dis_error_fake
                
                generator_error = generator_error + gen_error
            
            print("========== Epoch : {} | Generator Loss : {} | Discriminator Loss : {} =========="\
              .format(epoch+1, generator_error.detach(), discriminator_error.detach()))
        return self.generator, self.discriminator
