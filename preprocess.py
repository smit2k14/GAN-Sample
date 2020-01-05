import numpy as np
import pandas as pd
import PIL
from PIL import Image
import cv2


def Normalize(image,mean = [0.485,0.456,0.406],std = [0.229,0.224,0.225]):
    for channel in range(3):
        image[:,:,channel]=(image[:,:,channel]-mean[channel])/std[channel]
    return image

def DeNormalize(image, mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225]):
    for channel in range(3):
        image[:,:,channel] = image[:,:,channel]*std[channel]+mean[channel]
    return image