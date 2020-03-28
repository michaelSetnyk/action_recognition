'''
THE MODEL
'''
from feature_extractors import scn,tcn,stacn
from loaders import flownet2
import torch 
from flownet2.models import FlowNet2
from torch.utils.data import DataLoader
import torch.utils.data as data
from torchvision.io import read_video
import torchvision.transforms.functional as F
from torchvision import transforms
import os, math, random
from os.path import *
import numpy as np
from imageio import imread
from torch.autograd import Variable
from sklearn.cluster import KMeans

from glob import glob

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def crop(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]


class SingleVideoDataset(data.Dataset):
    def __init__(self,args,path):
        render_size = args.inference_size
        
        frames,_,_ = read_video(path,pts_unit='sec')
        frames= frames.to(torch.float32)

        # Step 1 crop the frames
        frame_size = frames[0].shape
        
        if (render_size[0] < 0) or (render_size[1] < 0) or (frame_size[0]%64) or (frame_size[1]%64):
            render_size[0] = ( (frame_size[0])//64 ) * 64
            render_size[1] = ( (frame_size[1])//64 ) * 64
   
        cropped_frames = []
        for frame in frames:
            image_size = frame.shape[:2]
            cropper = StaticCenterCrop(image_size, render_size)
            frame  = cropper.crop(frame)
            frame = frame.permute(2, 0, 1)
            cropped_frames.append(frame)
        self.frames = [torch.stack(cropped_frames)]
    
    def __len__(self):
        return len(self.frames)
    def __getitem__(self,idx):
        frame = self.frames[idx]
        return frame

def predict(args,video_path):
    testset = SingleVideoDataset(args,video_path)
    test_loader = DataLoader(testset,batch_size=1,shuffle=False)
    features = []
    model = ActionS(args)
    for video in test_loader:
        with torch.no_grad():
            model(video).cpu()
        
class ActionS(torch.nn.Module):
    def __init__(self,args):
        super(ActionS,self).__init__()
        self.scn = scn.model(args)
        self.flownet2 = flownet2.model(args)
        if args.cuda:
            self.flownet2.cuda()
        self.flownet2.eval()
        self.args = args
        #self.tcn = tcn.model(args)
        #self.stacn = stacn.model(args)

    def forward(self,inputs):
        inputs = torch.squeeze(inputs)
        #Precompute the flow 
        #f = self._precompute_flow(inputs)
        
        spatio_features =  self.scn(inputs)
        self._avfs(spatio_features)
        return spatio_features


    def _avfs(self,features):
        k = 8
        flat_features = [f.flatten() for f in features.cpu().numpy()]
        # Step 1 construct the Codebook C
    
        C = KMeans(n_clusters=k,init='k-means++',tol=0.0001).fit(flat_features)




    def _composite_flow(i1,i2):
        '''
        These are two rgb frames non neighbour frames 
        The paper tells us to use the method described in 
        #http://www.cs.uu.nl/groups/MG/multimedia/publications/art/PR2016.pdf
        for the sake of time we will re-calculate flow.
        Bad for efficiency good for time and complexity
        '''
        return _precompute_flow([i1,i2])

    def _precompute_flow(self,inputs):
        flow=[]
        with torch.no_grad():
            for i in range(len(inputs)-1):
                flow_image = [inputs[i],inputs[i+1]]
                if self.args.cuda:
                    flow_image = [d.cuda(non_blocking=True) for d in flow_image]
                flow_image = [Variable(d) for d in flow_image]
                flow_image = flow_image[0].unsqueeze(0)
                f=self.flownet2(flow_image).cpu()
                flow.append(f)
        return flow
