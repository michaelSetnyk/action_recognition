from numpy import dot
from numpy.linalg import norm
from scipy import spatial
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from loaders.flownet2 import flow_from_frames
import torchvision.transforms.functional as F
import torch
import cv2
'''
Deep features of each frame
F: F{f1,f2..fn..fN}

k means to learn a Codebook C withh a vocabulary of K visual words 
C = {c1,c2,cK}

Actions 
S = {S1..Sn}
'''

def CS(fsk,fi):
    result = 1 - spatial.distance.cosine(fsk, fi)
    return result 

'''
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
def AVFS(f):
    print("AVFS")
    k = 8
    tau = 0.85
    
    ff = [fi.flatten() for fi in f]
    # Step 1 construct the Codebook C
    
    C = KMeans(n_clusters=k,init='k-means++',tol=0.0001).fit(ff)
    
    # Step 2 Construct ActionS 
    S = []

    # set of indexes of key frames 
    fsk = [f[0]]
    flat_fsk=[ff[0]]

    Si = [f[0]]
    fski = fsk[0]
    flat_fski = ff[0]
    
    # Determine if features maps from f2 to fi belong to fsk1 (S[1])
    for i in range(1,len(f)):
        fi = f[i]
        flat_fi = ff[i]
        sim = CS(flat_fski,flat_fi)
        # end of the current action 
        if sim < tau:
            print("NEW ACTION")
            S.append(Si)
            Si = []
            fski = fi
            flat_fski = flat_fi
            fsk.append(fski)
            flat_fsk.append(flat_fski)
        Si.append(fi)
    S.append(Si)
    
    [print(f"S[{i}] = {len(S[i])}") for i in range(len(S))]
    
    # Step 3 segmentation updating 
    # update segments around the key frames 
    i = len(S[0])
    for s_i in range(1,len(S)):
        prev_label = C.labels_[i-1]
        Si = S[s_i]
        for j in range(len(Si)):
            index = i +j
            label = C.labels_[index]
            # Update segments 
            if label == prev_label:
                feature_map = Si[j]
                S[s_i-1].append(feature_map)
                S[s_i].pop(0)
                if not S[s_i]:
                    S.pop(s_i)
                    fsk.pop(s_i)
            else:
                #make sure that the key frame exists in the actions segment
                fsk[s_i] = S[s_i][0]
                break
        # End of current set 
        i+=len(Si)

    [print(f"S[{i}] = {len(S[i])}") for i in range(len(S))]
    # Maybe update the fsk to make sure the feature is in the set 
    return S,fsk 

'''
Adaptive Segment Feature Sampling
'''
'''
def SM(i,j):
    fskj = fsk[j]
    fsi_j = bilinear_warp(fsi,flow(i,j))


    top = exp(- mag(warped - fskj   )**2)
    bommot_expr = exp(- mag(     - fskj)**2)
'''


'''
flow: resized flow field (7,7,1024)
features: feature map (7,7,1024)

p: current location in feature_map
s_p: current_location in flow

G: the kernel
'''
def warp(key_frame,flow,feature_map):
    assert key_frame.shape == feature_map.shape and flow.shape ==(7,7,2) and feature_map.shape == (7,7,1024), "shapes do not match" 
    def G(q,p):
        # current location in the feature map
        # s_p is each x y location in the flow map 
        # q is each x y location in the feature map
        def g(a,b):
            return np.max([0.0,1.0 - abs(a-b)])

        g1 = g(q[0],p[0])
        g2 = g(q[1],p[1])
        return g1 * g2

    warped = np.empty(key_frame.shape)
    for y in range(7):
        for x in range(7):
            q =(x,y)
            p = q 
            s_p = flow[p]
            G_val = G(q,p+s_p)
            warped[q] = G_val*key_frame[q]
    
    return warped

def ASFS(args,rgb_frames,S,fsk):
    print("ASFS")
    rgb_start_index = 0
    for s_i,Si in enumerate(S):
        #get the refrence rgb_frame and feature map 
        key_frame = fsk[s_i]
        refrence_frame = rgb_frames[rgb_start_index]
        for j in range(1,len(Si)):
            frame_index = rgb_start_index + j
            feature_map = Si[j]
            frame = rgb_frames[rgb_start_index]
            
            flow_images = [refrence_frame,frame]
            flow = flow_from_frames(args,flow_images)[0].transpose(1,2,0)
            flow = cv2.resize(flow,(7,7))
            warp(key_frame,flow,feature_map)
            #SM(warped,refrence_frame)
    





