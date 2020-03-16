import torch
import torch.utils.data as data
from torchvision.io import read_video
import random
import numpy as np
from loaders.flownet2 import flow_from_frames

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

def MCM(flow):
    flow  =  flow.transpose(1,2,0)
    def mag(uv):
        return np.sqrt(uv.dot(uv))

    mcm = np.apply_along_axis(mag,2,flow) 

    max_mag = np.max(mcm)
    min_mag = np.min(mcm)
    mcm = mcm - min_mag
    mcm = mcm/(max_mag-min_mag)
    return mcm

def  RGBF(rgb_vec,mcm):
    def rgbf(vec):
        pixels = vec[:3]
        mag = vec[3]
        return pixels * mag
    s = np.dstack((rgb_vec,mcm))
    rgbf_vec = np.apply_along_axis(rgbf,2,s)
    return rgbf_vec


class StacnVideoFromFolder(data.Dataset):
  def __init__(self, args, is_cropped, path = '/path/to/frames/only/folder'):
    self.frames = load_rgbf(args,is_cropped,path)
     

  def __getitem__(self, index):
    img = self.frames[index] 
    img = img.resize_(3,224,224)
    
    return img.cuda()

  def __len__(self):
    return len(self.frames)


def load_rgbf(args,is_cropped,path):
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

    rgb_frames = cropped_frames

    # Step 2 load the flow images 
    flow_frames = flow_from_frames(args,rgb_frames)
    frames = []    
    # Step 3 create the rgbf images 
    
    for (index, flow_img) in enumerate(flow_frames):
        flow_img = flow_img.cpu().numpy()  
        mcm = MCM(flow_img)
        rgb_img = rgb_frames[index]
        rgbf = RGBF(rgb_img,  mcm)
        frames.append(rgbf)
    return frames

def load_rgbf_from_frames(args,is_cropped,rgb_frames):
    render_size = args.inference_size

    # Step 2 load the flow images 
    flow_frames = flow_from_frames(args,rgb_frames)
    frames = []    
    # Step 3 create the rgbf images 
    
    for (index, flow_img) in enumerate(flow_frames):
        flow_img = flow_img.cpu().numpy()  
        mcm = MCM(flow_img)
        rgb_img = rgb_frames[index]
        rgbf = RGBF(rgb_img,  mcm)
        frames.append(rgbf)
    return frames

