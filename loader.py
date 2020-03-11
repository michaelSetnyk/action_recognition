import torch
import torch.utils.data as data
from torchvision.io import read_video
import os, math, random
from os.path import *
import numpy as np
from imageio import imread

from glob import glob

class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]

class VideoFromFolder(data.Dataset):
  def __init__(self, args, is_cropped, path = '/path/to/frames/only/folder', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates
    
    self.frames=[]
    frames,_,_ = read_video(path,pts_unit='sec')
    frames= frames.to(torch.float32)
    for i in range(len(frames)-1):
        im1 = frames[i]
        im2 = frames[i+1]
        self.frames += [ [ im1, im2 ] ]

    self.size = len(self.frames)
    self.frame_size = self.frames[0][0].shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = self.frames[index][0]
    img2 = self.frames[index][1]
    
    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
   
    images = torch.stack(images)
    images = images.permute(3,0, 1, 2)
    return [images], [torch.zeros(images.size()[0:1] + (2,) + images.size()[-2:])]

  def __len__(self):
    return self.size * self.replicates
