import os, sys
import torch
import numpy as np
from  torchsummary import summary
import torchvision
from torchvision import models
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2
from flownet2.utils.flow_utils import flow2img
from flownet2 import models
from feature_extractors import scn,tcn,stacn
from loaders.flownet2 import flow_images
from loaders.stacn import load_rgbf
import cluster
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument("--rgb_max", type=float, default=255.)
parser.add_argument('--crop_size', type=int, nargs='+', default = [256, 256], help="Spatial dimens    ion to crop training samples for training")
parser.add_argument('--inference_size', type=int, nargs='+', default = [-1,-1], help='spatial size     divisible by 64. default (-1,-1) - largest possible valid size would be used')
parser.add_argument('--flow_vis', action='store_true', help='display flow')
args = parser.parse_args()      

args.__dict__["cuda"]= True if torch.cuda.is_available() else False
# get flow images
test_path = "hmdb51/dribble/10YearOldYouthBasketballStarBaller_dribble_f_cm_np1_ba_med_5.avi"
test_path = "walk/ira_walk.avi"
#flow_vecs = flow_images(args,test_path)
scn_vecs = scn.features(args,test_path)
#tcn_vecs = tcn.features(args,test_path)
#stacn_vecs = stacn.features(args,test_path)

cluster.AVFS(scn_vecs)
