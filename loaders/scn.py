import torch
import torch.utils.data as data
from torchvision.io import read_video
import torchvision.transforms.functional as F
from torchvision import transforms
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

    def drop(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def crop(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]

class ScnVideoFromFolder(data.Dataset):
  def __init__(self, args, path = '/path/to/frames/only/folder'):
    self.args = args
    self.render_size = args.inference_size
    self.frames,_,_ = read_video(path,pts_unit='sec')
    self.frames= self.frames.to(torch.float32)
    self.size = len(self.frames)
    self.frame_size = self.frames[0][0].shape
    args.inference_size = self.render_size

  def __getitem__(self, index):
    img = self.frames[index] 
    img = img.resize_(3,224,224)
    
    return img.cuda() 

  def __len__(self):
    return len(self.frames)

def load_rgb(args,is_cropped,path):
    render_size = args.inference_size
    
    rgb_frames,_,_ = read_video(path,pts_unit='sec')
    rgb_frames= rgb_frames.to(torch.float32)
    # Step 1 crop the images
    frame_size = rgb_frames[0].shape
    if (render_size[0] < 0) or (render_size[1] < 0) or (frame_size[0]%64) or (frame_size[1]%64):
        render_size[0] = ( (frame_size[0])//64 ) * 64
        render_size[1] = ( (frame_size[1])//64 ) * 64
   
    cropped_frames = []
    for rgb_frame in rgb_frames:
        image_size = rgb_frame.shape[:2]
        if is_cropped:
            cropper = StaticRandomCrop(image_size, crop_size)
        else:
            cropper = StaticCenterCrop(image_size, render_size)
        frame  = cropper.crop(rgb_frame)
        cropped_frames.append(frame)

    return cropped_frames

