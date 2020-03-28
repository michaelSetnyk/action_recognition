import os, sys
import torch
from torch import nn
import numpy as np
from torchsummary import summary
from torch.utils.data import DataLoader
from loaders.flownet2 import flow_images
from loaders.tcn import TcnStacksFromFlow

def model(args):
    in_channels = 20
    base_model = torch.hub.load('pytorch/vision:v0.5.0', 'googlenet', pretrained=True)
    #should actually use this but we don't have the checkpoint file
    #base_model = bninception.BNInception()
    modules = list(base_model.modules())
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]
    params = [x.clone() for x in conv_layer.parameters()]
    kernel_size = params[0].size()
    new_kernel_size = kernel_size[:1] + (in_channels, ) + kernel_size[2:]
    new_kernel_data = params[0].data.mean(dim=1, keepdim=True).expand(
    new_kernel_size).contiguous()  # make contiguous!
    new_conv_layer = nn.Conv2d(in_channels, conv_layer.out_channels,conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,bias=True if len(params) == 2 else False)
    new_conv_layer.weight.data = new_kernel_data
    if len(params) == 2:
        new_conv_layer.bias.data = params[1].data
        # remove ".weight" suffix to get the layer layer_name
    layer_name = list(container.state_dict().keys())[0][:-7]
    setattr(container, layer_name, new_conv_layer)
    if args.cuda:
        tcn = container.cuda()
    return tcn 
    
def features(args,path):
    tcn = model(args)
    tcn.eval()
   
    images = flow_images(args,path)
    testset = TcnStacksFromFlow(images)
    test_loader = DataLoader(testset,batch_size=1,shuffle=False)
    features = []
    for stack in test_loader: 
        features.append(tcn(stack))

    return features
