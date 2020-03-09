import os, sys
import torch
import numpy as np
from  torchsummary import summary
from torchvision import models
from flownet2 import models
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument("--rgb_max", type=float, default=255.)
args = parser.parse_args()      

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# create scn 
print(device)
model = torch.hub.load('pytorch/vision:v0.5.0', 'googlenet', pretrained=True)
model.to(device)
scn = torch.nn.Sequential(*(list(model.children())[:-3]))
summary(scn,(3,224,224))
flownet2 = models.FlowNet2(args).cuda()
print(flownet2)
