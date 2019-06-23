# -*- coding: utf-8 -*-
"""GAN Sample Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Vvn40oMC6gaD0SvCyGk5lUo2aNuqiOD3
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import PIL
from PIL import Image
import cv2
import os, os.path
import matplotlib.pyplot as plt
# %matplotlib inline
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from google.colab import drive
drive.mount('/content/gdrive', force_remount = True)

def Normalize(image,mean = [0.485,0.456,0.406],std = [0.229,0.224,0.225]):
    for channel in range(3):
        image[:,:,channel]=(image[:,:,channel]-mean[channel])/std[channel]
    return image

def DeNormalize(image, mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225]):
    for channel in range(3):
        image[:,:,channel] = image[:,:,channel]*std[channel]+mean[channel]
    return image

imageDir = '/content/gdrive/My Drive/celeb_images/Part 1'

image_path_list = []
for file in os.listdir(imageDir):
    image_path_list.append(os.path.join(imageDir, file))

image = np.empty([len(image_path_list), 32, 32, 3])
for indx, imagePath in enumerate(image_path_list):
    if(indx>5000):
        break
    im = Image.open(imagePath).convert('RGB')
    im = im.resize((32, 32))
    im=np.array(im,dtype=np.float32)
    im=im/255
    im=Normalize(im, [0.5,0.5,0.5], [0.5,0.5,0.5])
    image[indx,:,:,:] = im

image = image[:5000,:,:,:]

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.seq1 = nn.Sequential(
                    nn.Linear(100, 32768),
                    nn.BatchNorm1d(32768)
                    )
        self.seq2 = nn.Sequential(
                    nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 5, stride = 1, padding = 2),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU()
        )
        self.seq3 = nn.Sequential(
                    nn.ConvTranspose2d(in_channels = 128, out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
                    nn.LeakyReLU()
        )
        self.seq4 = nn.Sequential(
                    nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 5, stride = 1, padding = 2),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU()
        )
        self.seq5 = nn.Sequential(
                    nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 5, stride = 1, padding = 2),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU()
        )
        self.seq6 = nn.Sequential(
                    nn.Conv2d(in_channels = 128, out_channels = 3, kernel_size = 5, stride = 1, padding = 2),
                    nn.BatchNorm2d(3),
                    nn.LeakyReLU()
        )
    def feed_forward(self, inp):
        x = self.seq1(inp).view(inp.shape[0], 128, 16, 16)
        x = self.seq2(x)
        x = self.seq3(x)
        x = self.seq4(x)
        x = self.seq5(x)
        x = self.seq6(x)
        return x

gen = Generator()
gen.feed_forward(Variable(torch.randn(2, 100)))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.seq1 = nn.Sequential(
                    nn.Conv2d(in_channels = 3, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU()
        )
        self.seq2 = nn.Sequential(
                    nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU()
        )
        self.seq3 = nn.Sequential(
                    nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU()
        )
        self.seq4 = nn.Sequential(
                    nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU()
        )
        self.seq5 = nn.Sequential(
                    nn.Dropout(0.4),
                    nn.Linear(4*4*128, 1),
                    nn.Sigmoid()
        )
    def feed_forward(self, x):
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        x = self.seq4(x).reshape(x.size(0), -1)
        x = self.seq5(x)
        return x

class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
        self.generator = Generator().float().cuda()
        self.discriminator = Discriminator().float().cuda()
        self.generator_optim = optim.Adam(self.generator.parameters())
        self.discriminator_optim = optim.Adam(self.discriminator.parameters())
        
    def generate_images(self, inp):
        out = self.generator.feed_forward(inp)
        return out
    
    def train_gen(self, loss, batch_size):
        noise = Variable(torch.randn(batch_size, 100)).float().cuda()
        gen_images = self.generator.feed_forward(noise)
        fake_images_prediction = self.discriminator.feed_forward(gen_images.cuda())
        fake_images_truth = Variable(torch.ones(batch_size)).float().cuda()
        dis_error_fake = loss(fake_images_prediction, fake_images_truth)
        dis_error_fake.backward()
        self.generator_optim.step()
        generator_error = dis_error_fake
        return generator_error
    
    def train_dis(self, loss, batch_size, orig_images):
        fake_images_truth = Variable(torch.ones(batch_size)).float().cuda()
        noise = Variable(torch.randn(batch_size, 100)).float().cuda()
        gen_images = self.generator.feed_forward(noise).cuda()
        fake_images_prediction = self.discriminator.feed_forward(gen_images)
        dis_error_fake = loss(fake_images_prediction, fake_images_truth)
        true_images_truth = Variable(torch.zeros(len(orig_images))).float().cuda()
        true_images_prediction = self.discriminator.feed_forward(orig_images)
        dis_error_true = loss(true_images_prediction, true_images_truth)
        discriminator_error = dis_error_fake + dis_error_true
        return dis_error_fake, dis_error_true
    
    def train(self, images, epochs = 20):
        print('============ Starting training of GAN ============')
        batch_size = 100
        loss = nn.BCELoss()
        for epoch in range(epochs):
            discriminator_error = 0
            generator_error = 0
            self.generator_optim.zero_grad()
            self.discriminator_optim.zero_grad()
            for i in range(int(len(images)/batch_size) + 1):
                try:
                    orig_images = torch.from_numpy(images[i*batch_size:(i+1)*batch_size, :, :, :]).permute(0, 3, 1, 2).float().cuda()
                    if(orig_images.shape[0]==0):
                        break
                except:
                    orig_images = torch.from_numpy(images[i*batch_size:]).permute(0, 3, 1, 2).float()
                
                #Training the discriminator
                dis_error_fake, dis_error_true = self.train_dis(loss, batch_size, orig_images)
                if(dis_error_fake+dis_error_true>1 or epoch>20):
                    dis_error_fake.backward()
                    dis_error_true.backward()
                    self.discriminator_optim.step()
                discriminator_error = discriminator_error + dis_error
                
                #Training the generator
                gen_error = self.train_gen(loss, batch_size)
                generator_error = generator_error + gen_error
            print("========== Epoch : {} | Generator Loss : {} | Discriminator Loss : {} =========="\
              .format(epoch+1, generator_error.detach(), discriminator_error.detach()))
            
            
            if((epoch+1)%10 == 0):
                
                noise = Variable(torch.randn(3, 100)).float().cuda()
                gen_images = self.generator.feed_forward(noise)
                count = 0
                for gen_image in gen_images:
                    img = Image.fromarray(np.uint8((DeNormalize(gen_image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])*255).permute(2, 1, 0).squeeze(0).cpu().detach()))
                    img.save('Epoch {} count {}.jpg'.format(epoch+1, count+1))
                    count+=1
                    #plt.imshow(DeNormalize(gen_image).permute(0, 3, 2, 1).squeeze(0).cpu().detach())
        return self.generator, self.discriminator

gan = GAN()
gen, dis = gan.train(image,200)

noise = Variable(torch.randn(1, 16, 2, 2)).float().cuda()
im = gen.feed_forward(noise)
im = im.permute(0, 2, 3, 1)
im = im.squeeze(0).cpu().detach().numpy()*255
im = DeNormalize(im,[0.485,0.456,0.406],[0.229,0.224,0.225])

plt.imshow(im)

gen, dis = gan.train(image,500)
