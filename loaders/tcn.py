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

class TcnStacksFromFlow(data.Dataset):
  def __init__(self,images):
    self.stacks = []
    for i in range(len(images)-9):
        tensors = [torch.tensor(images[j+i]).cuda()  for j in range(0,10) ]
        stack = torch.cat(tensors,0)
        self.stacks.append(stack)
    print(len(self.stacks))

  def __getitem__(self, index):
    stack = self.stacks[index] 
    return stack 

  def __len__(self):
    return len(self.stacks)
