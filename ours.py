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
import torch.nn.functional as NNFunc
from torchvision import transforms
import os, math, random
from os.path import *
import numpy as np
import numpy.linalg as LA
from imageio import imread
from torch.autograd import Variable
from sklearn.cluster import KMeans
import imageio
from scipy import spatial
from glob import glob
import cv2 
from flownet2.utils.flow_utils import flow2img
import kornia

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def crop(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]


class SingleVideoDataset(data.Dataset):
    def __init__(self,args,path):
        reader = imageio.get_reader(path, 'ffmpeg')
        l = int(reader.count_frames()) -1
        frames = []
        for i in range(l):
            frame = reader.get_data(i)
            frame = torch.from_numpy(frame).to(torch.float32)
            frames.append(frame) 
        frames =  [torch.stack(frames)]
        self.frames = frames

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
        if args.cuda:
            video = video.cuda()
        with torch.no_grad():
            model(video).cpu()
        
class ActionS(torch.nn.Module):
    def __init__(self,args):
        super(ActionS,self).__init__()
        self.args = args
        
        flowNetPath = "flownet2/FlowNet2_checkpoint.pth.tar"
        flowNetCheckpoint = torch.load(flowNetPath)
        state_dict = flowNetCheckpoint["state_dict"]
        flownet2 = FlowNet2(args)
        
        self.scn = scn.model(args)
        if args.cuda:
            flownet2 = flownet2.cuda()
            self.scn.cuda()
        
        flownet2.load_state_dict(state_dict)
        self.flownet2 = flownet2
        self.flownet2.eval()
        self.scn.eval()
        #self.tcn = tcn.model(args)
        #self.stacn = stacn.model(args)

    def forward(self,frames):
        frames =  torch.squeeze(frames)
        self.segments = []
        print(frames.shape)
        self.frames = frames 
        self.flow_frames =  self._precompute_flow(frames)

        self.rgb_frames = self._rgb_frames(frames)
        
        spatio_features = self.scn(self.rgb_frames)
        self._avfs(spatio_features)
        self._asfs(spatio_features)

        return spatio_features

    def _rgb_frames(self,frames):
        rgb_frames = []
        for frame in frames:
            frame = frame.resize_(3,224,224)
            rgb_frames.append(frame)
        return torch.stack(rgb_frames)

    def _CS(self,fsk,fi):
        result = 1 - spatial.distance.cosine(fsk, fi)
        return result 
    
    '''
    Can probably convert this into a pure PyTorch implementation

    Adaptive Video Feature Segmentation
    C: codebook found by computing K means clustering
    Sk: The set of segments S
    S:  The set of features within an action Set = {f1..fi}
    fsk: The set of key frames 
    fski: The key frame of a set, each set has 1 key frame, this is the first frame 

    Steps 
    1) Compute the codebook
    2) Determine the intial actions
    3) Segmentation update 
    '''
    def _avfs(self,features):
        tau  = 0.85
        k = 64
        flat_features = [f.cpu().numpy().transpose(2,1,0).flatten() for f in features]
        # Step 1 construct the Codebook C
        C = KMeans(n_clusters=k,init='k-means++',tol=0.0001).fit(flat_features)
    
        # Step 2 Construct ActionS,key_feature_maps,feature_maps
        segments = []

        
        # Determine if features maps from f2 to fi belong to fsk1 (S[1])
        key = 0
        S = [key]
        for i in range(1,len(features)):
            # end of the current action 
            sim = self._CS(flat_features[key],flat_features[i])
            if sim < tau:
                # put some length assets here
                segment = {"key":key,
                            "indexes":S
                            }
                segments.append(segment)
                key = i
                S=[key]
            else:
                S.append(i)

        segment = {"key":key,
                "indexes":S} 
        segments.append(segment)
        [print(f"Sk[{i}] = {len(segments[i]['indexes'])}") for i in range(len(segments))]
        
        # Step 3 segmentation updating 
        # update segments around the key frames 
        for seg_index in range(1,len(segments)):
            previous_segment = segments[seg_index-1]
            current_segment = segments[seg_index]
            prev_label = C.labels_[previous_segment["indexes"][-1]]
            for j in segment["indexes"]:
                label = C.labels_[j]
                # Update segments 
                if label == prev_label:
                    previous_segment["indexes"].append(j)
                    segment["indexes"].pop(0) 
                    # check if the list is now empty
                    if not segment["indexes"]:
                        segments.pop(seg_index)
            # End of current set 

        [print(f"Sk[{i}] = {len(segments[i]['indexes'])}") for i in range(len(segments))]
        self.segments = segments

    
    def _precompute_flow(self,frames):
        '''
        for some reason flownet demands that height and width do not match
        '''
        # Step 1 crop the frames
        frame_size = frames[0].shape
        render_size = self.args.inference_size
        
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
        inputs = cropped_frames  
        flow=[]
        
        for i in range(len(inputs)-1):
            flow_input = [inputs[i],inputs[i+1]]
            flow_input = torch.stack(flow_input)
            flow_input = flow_input.permute(1,0,2,3)
            if self.args.cuda:
                flow_input = [d.cuda(non_blocking=True) for d in flow_input]
            flow_input = [Variable(d) for d in flow_input]
            flow_input = torch.stack(flow_input).unsqueeze(0)
            flow_input = flow_input
            with torch.no_grad():
                # NEED SHAPE[1,3,2,192,320]
                flow_image=torch.squeeze(self.flownet2(flow_input))
                if self.args.flow_vis:
                    img = flow2img(flow_image.cpu().numpy().transpose(1,2,0)).astype(np.uint8)
                    cv2.imshow("flow image",img)
                    cv2.waitKey(0)

                flow.append(flow_image)
        return flow 


    '''
    Adaptive Segment Feature Sampling
    j: is the key frame index 
    i: is the refrence_frame index 
    '''
    def _sm(self,i,j,idx,warps, key_frame):
        def expr(w,k):
            d = np.subtract(w,k)
            n = LA.norm(d)
            p = np.power(n,2)
            top = np.exp(np.negative(p))
            return top

        top = expr(warps[idx],key_frame)
        bot = np.copy(top)
        #for a in range (idx):
        #    bottom  = np.add(bottom,expr(warps[a],key_frame))
        #sim = np.divide(top,bottom)
        sim = top/bot
        if np.isnan(sim):
            return 0.0

        return sim


    '''
    flow: resized flow field (7,7,1024)
    features: feature map (7,7,1024)

    p: current location in feature_map
    s_p: current_location in flow

    G: the kernel
    '''
    def _warp(self,key_features,flow):

        assert key_features.shape == (7,7,1024),   "key frame does not have shape (7,7,1024)"
        assert flow.shape ==(7,7,2),             "flow does not have shape (7,7,2)" 
        def G(q,shift):
            # current location in the feature map
            # s_p is each x y location in the flow map 
            # q is each x y location in the feature map
            def g(a,b):
                d = np.subtract(a,b)
                v = np.subtract(1.0,np.abs(d))
                result = np.max(np.array([0.0,v]))
                return result

            qx,qy = q[0],q[1]
            px,py = shift[0],shift[1]

            v = np.dot(g(qx,px),g(qy,py))
            return v

        def channel_warp(p):         
            s = np.zeros((1024))
            for y in range(7):
                for x in range(7):
                    q = (x,y)
                    shift = p + flow[q]
                    s += G(q,shift)*key_features[q]
            warped_features = s
            return warped_features

        warped = np.zeros((7,7,1024))
        for y in range(7):
            for x in range(7):
                p = (x,y)
                w = channel_warp(p)
                warped[p] = w 
        return warped

    def _composite_flow(self,j,i):
        new_flow= self.flow_frames[j].clone()
        for index in range(j+1,i):
            v = self.flow_frames[index][0]
            new_flow += v
        return new_flow

    # speed up by doing flow all at once 
    def _asfs(self,features):
        '''
        j is the key frame index 
        i is the feature index

        F(i,j)
        F(i,i+1)
        '''
        print("ASFS")
        tau2 = torch.Tensor([0.75]).cuda()
        tau3 = torch.Tensor([0.04]).cuda()

        flow_frames = self.flow_frames
        for segment in self.segments:
            '''
            It appears that the paper is off on this example
            they assume key is last frame but it's usually the first frame 
            '''

            j = segment["key"]
            indexes = segment["indexes"]
            warps = []
            for idx,i in enumerate(indexes):
                if j+1 == i:
                    flow = flow_frames[j]
                else:
                    flow = self._composite_flow(j,i)

                #flow = NNFunc.interpolate(flow,(7,7),mode="bilinear",align_corners=False)
                #flow = torch.squeeze(flow)
                flow = torch.squeeze(flow)
                flow = flow.cpu().numpy().transpose(2,1,0)
                
                flow = cv2.resize(flow,(7,7))
                # Next step is to apply the warping function to the flow
                key_features = features[j].permute(2,1,0).cpu().numpy()
                feature_map = features[i].permute(2,1,0).cpu().numpy()
                
                warped = self._warp(key_features,flow)
                warps.append(warped)
               
                print("before sm warp: ",np.max(warped))
                print("before sm key: ",np.max(key_features))
                sim = self._sm(i,j,idx,warps,key_features)
                
                if sim>tau2:
                    # COND 1 if frames are too similar remove local feature
                    print("SIMILAR")
                elif sim<tau3:
                    # COND  2 if frames are too disimilar then discard the frame and it's features
                    print("NOISY FRAME")
        # Pool the weight for each segment
        fvsk = [] 
        for j,Si in enumerate(self.add_module):
            sims = similarties[j]
            print(len(sims),len(Si))
            assert len(sims) == len(Si), "similarties and length of Si do not match"

            avg = 1.0/(np.sum(sims))
            w = []
            for i in range(len(Si)):
                fsi = Si[i]
                s = sims[i]
                w.append(s*fsi)
            v = np.sum(w) * avg
            fvsk.append(v)
        print(fvsk)
        # Now take the L2  norm 
        LA.norm(fvsk)
