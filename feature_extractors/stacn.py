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
from loaders.stacn import StacnVideoFromFolder

def model(args):
    base_model = torch.hub.load('pytorch/vision:v0.5.0', 'googlenet', pretrained=True)
    if  args.cuda:
        base_model.cuda()
    stacn = torch.nn.Sequential(*(list(base_model.children())[:-3]))
    return stacn
# get stacn features  
def features(args,video_path):
     
    stacn = model(args)
    stacn.eval()
    testset = StacnVideoFromFolder(args,False,video_path)
    test_loader = DataLoader(testset,batch_size=1,shuffle=False,num_workers=4)
    
    features = []
    for image in test_loader:
        with torch.no_grad():
            features.append(stacn(image))
        
    return features
