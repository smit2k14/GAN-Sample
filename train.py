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
from models import Generator, Discriminator, GAN
from preprocess import Normalize, DeNormalize

if __name__ == '__main__':
    imageDir = '...celeb_images/Part 1'

    image_path_list = []
    for file in os.listdir(imageDir):
        image_path_list.append(os.path.join(imageDir, file))

    image = np.empty([len(image_path_list), 64, 64, 3])
    for indx, imagePath in enumerate(image_path_list):
        if(indx>5000):
            break
        im = Image.open(imagePath).convert('RGB')
        im = im.resize((64, 64))
        im=np.array(im,dtype=np.float32)
        im=im/255
        im=Normalize(im, [0.5,0.5,0.5], [0.5,0.5,0.5])
        image[indx,:,:,:] = im

    image = image[:5000,:,:,:]


    gan = GAN()
    gen, dis = gan.train(image,200)

    noise = Variable(torch.randn(1, 100, 1, 1)).float().cuda()
    im = gen.feed_forward(noise)
    im = im.permute(0, 2, 3, 1)
    im = im.squeeze(0).cpu().detach().numpy()*255
    im = DeNormalize(im,[0.5,0.5,0.5],[0.5,0.5,0.5])

    plt.imshow(im)