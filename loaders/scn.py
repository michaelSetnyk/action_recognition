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
