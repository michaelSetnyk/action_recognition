import torch
import torch.utils.data as data
from torchvision.io import read_video
import os, math, random
from os.path import *
import numpy as np
from imageio import imread
from torch.utils.data import DataLoader
from flownet2.models import FlowNet2
from flownet2.utils.flow_utils import flow2img
import cv2
from torch.autograd import Variable

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

class FlowVideoFromFolder(data.Dataset):
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


class FlowVideoFromFrames(data.Dataset):
  def __init__(self, args, rgb_frames, replicates = 1):
    self.args = args
    self.replicates = replicates
    
    self.frames= []
    for i in range(len(rgb_frames)-1):
        im1 = rgb_frames[i]
        im2 = rgb_frames[i+1]
        self.frames += [ [ im1, im2 ] ]

    self.size = len(self.frames)

  def __getitem__(self, index):
    index = index % self.size

    img1 = self.frames[index][0]
    img2 = self.frames[index][1]
    
    images = [img1, img2]
   
    images = torch.stack(images)
    images = images.permute(3,0, 1, 2)
    return [images], [torch.zeros(images.size()[0:1] + (2,) + images.size()[-2:])]

  def __len__(self):
    return self.size * self.replicates

def flow_from_frames(args,frames):
    flowNetPath = "flownet2/FlowNet2_checkpoint.pth.tar"
    flownet2 = FlowNet2(args)
    
    if args.cuda:
        flownet2.cuda()
    
    flowNetCheckpoint = torch.load(flowNetPath)
    state_dict = flowNetCheckpoint["state_dict"]
    flownet2.load_state_dict(state_dict)
    flownet2.eval()
    
    testset = FlowVideoFromFrames(args,frames)
    test_loader = DataLoader(testset,batch_size=1,shuffle=False)

    flow_images = [] 
    for batch_idx, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = [d.cuda(non_blocking=True) for d in data], [t.cuda(non_blocking=True) for t in target]
        data, target = [Variable(d) for d in data], [Variable(t) for t in target]
        with torch.no_grad():
            outputs = flownet2(data[0])[0].cpu().numpy()
            flow_images.append(outputs)
            if args.flow_vis:
                img = flow2img(outputs.transpose(1,2,0)).astype(np.uint8)
                cv2.imshow("image",img)
                cv2.waitKey(0)
    return flow_images

def flow_images(args,path):
    flowNetPath = "flownet2/FlowNet2_checkpoint.pth.tar"
    flownet2 = FlowNet2(args)
    
    if args.cuda:
        flownet2.cuda()
    
    flowNetCheckpoint = torch.load(flowNetPath)
    state_dict = flowNetCheckpoint["state_dict"]
    flownet2.load_state_dict(state_dict)
    flownet2.eval()
    
    testset = FlowVideoFromFolder(args,False,path)
    test_loader = DataLoader(testset,batch_size=1,shuffle=False)

    flow_images = [] 
    for batch_idx, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = [d.cuda(non_blocking=True) for d in data], [t.cuda(non_blocking=True) for t in target]
        data, target = [Variable(d) for d in data], [Variable(t) for t in target]
        with torch.no_grad():
            outputs = flownet2(data[0])[0].cpu().numpy()
            flow_images.append(outputs)
            if args.flow_vis:
                img = flow2img(outputs.transpose(1,2,0)).astype(np.uint8)
                cv2.imshow("image",img)
                cv2.waitKey(0)
    return flow_images

