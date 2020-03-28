import os, sys
import torch
import numpy as np
from  torchsummary import summary
from torchvision import models
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2
from flownet2.utils.flow_utils import flow2img
from flownet2 import models
from loaders.flownet2 import FlowVideoFromFolder
from loaders.scn import ScnVideoFromFolder
from loaders.tcn import TcnStacksFromFlow

def model(args):
    base_model = torch.hub.load('pytorch/vision:v0.5.0', 'googlenet', pretrained=True)
    scn = torch.nn.Sequential(*(list(base_model.children())[:-3]))
    return scn
# get scn features  
def features(args,video_path):
    scn = model(args) 

    if  args.cuda:
        scn.cuda()
    
    scn.eval()
    
    testset = ScnVideoFromFolder(args,video_path)
    test_loader = DataLoader(testset,batch_size=1,shuffle=False)
    features = []
    for image in test_loader:
        with torch.no_grad():
            feature = scn(image).cpu().numpy().squeeze().transpose(1,2,0)
            features.append(feature)
        
    return features
