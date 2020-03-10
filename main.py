import os, sys
import torch
import numpy as np
from  torchsummary import summary
from torchvision import models
from flownet2.networks import FlowNetC
from flownet2 import models
import argparse
from loader import ImagesFromFolder
from torch.autograd import Variable
from torch.utils.data import DataLoader
flowNetPath = "flownet2/FlowNet2_checkpoint.pth.tar"

parser = argparse.ArgumentParser()
parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument("--rgb_max", type=float, default=255.)
parser.add_argument('--crop_size', type=int, nargs='+', default = [256, 256], help="Spatial dimens    ion to crop training samples for training")
parser.add_argument('--inference_size', type=int, nargs='+', default = [-1,-1], help='spatial size     divisible by 64. default (-1,-1) - largest possible valid size would be used')
args = parser.parse_args()      

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if cuda else 'cpu')
# create scn 
print(device)
#model = torch.hub.load('pytorch/vision:v0.5.0', 'googlenet', pretrained=True)
#model.to(device)
#scn = torch.nn.Sequential(*(list(model.children())[:-3]))

flownet2 = models.FlowNet2(args).cuda()
flowNetCheckpoint = torch.load(flowNetPath)
state_dict = flowNetCheckpoint["state_dict"]

flownet2.load_state_dict(state_dict)
flownet2.eval()

test_path="run/daria"
#Do not crop infernce images
testset = ImagesFromFolder(args,False,test_path)
test_loader = DataLoader(testset,batch_size=1,shuffle=False)


for batch_idx, (data, target) in enumerate(test_loader):
    if cuda:
        data, target = [d.cuda(non_blocking=True) for d in data], [t.cuda(non_blocking=True) for t in target]
    data, target = [Variable(d) for d in data], [Variable(t) for t in target]
    with torch.no_grad():
        print(data[0].shape)
        outputs = flownet2(data[0])

