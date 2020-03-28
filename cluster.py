from numpy import dot
from numpy.linalg import norm
from scipy import spatial
import numpy as np
from numpy import linalg as LA
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
def AVFS(f,frames):
    print("AVFS")
    k = 8
    tau = 0.85
    
    ff = [fi.flatten() for fi in f]
    # Step 1 construct the Codebook C
    
    C = KMeans(n_clusters=k,init='k-means++',tol=0.0001).fit(ff)
    
    # Step 2 Construct ActionS,key_feature_maps,feature_maps
    fsk = [] 
    V = []
    Sk = []

    # set of indexes of key frames 
    fski = f[0]
    flat_fski = ff[0]
    
    fsk.append(fski)
    fSi = [f[0]]
    Si = [frames[0]]
    # Determine if features maps from f2 to fi belong to fsk1 (S[1])
    for i in range(1,len(f)):
        frame = frames[i]
        fi = f[i]
        flat_fi = ff[i]
        sim = CS(flat_fski,flat_fi)
        # end of the current action 
        if sim < tau:
            print("NEW ACTION")
            fski = fi
            flat_fski = flat_fi
            
            V.append(Si)
            Sk.append(fSi)
            fsk.append(fski)
            Si = []
            fSi = []
        Si.append(fi)
        fSi.append(frame)
    V.append(Si)
    Sk.append(fSi)
    
    [print(f"Sk[{i}] = {len(Sk[i])}") for i in range(len(Sk))]
    
    # Step 3 segmentation updating 
    # update segments around the key frames 
    i = len(Sk[0])
    for s_i in range(1,len(Sk)):
        prev_label = C.labels_[i-1]
        Si = Sk[s_i]
        for j in range(len(Si)):
            index = i +j
            label = C.labels_[index]
            # Update segments 
            if label == prev_label:
                feature_map = Si[j]
                Sk[s_i-1].append(feature_map)
                Sk[s_i].pop(0)
                if not Sk[s_i]:
                    Sk.pop(s_i)
                    fsk.pop(s_i)
            else:
                #make sure that the key frame exists in the actions segment
                fsk[s_i] = Sk[s_i][0]
                break
        # End of current set 
        i+=len(Si)

    [print(f"Sk[{i}] = {len(Sk[i])}") for i in range(len(Sk))]
    # Maybe update the fsk to make sure the feature is in the set 
    return fsk,V,Sk 

'''
Adaptive Segment Feature Sampling
j: is the key frame index 
i: is the refrence_frame index 



'''
def SM(i,j,warped, key_frame):
    def expr(w,k):
        return np.exp(- LA.norm(w - k)**2)

    top = expr(warped[i],key_frame)
    bottom = np.sum([expr(a,key_frame) for a in range(i)])
    sim = top/bottom 
    return sim


'''
flow: resized flow field (7,7,1024)
features: feature map (7,7,1024)

p: current location in feature_map
s_p: current_location in flow

G: the kernel
'''
def warp(key_frame,flow,feature_map):
    assert key_frame.shape == (7,7,1024),   "key frame does not have shape (7,7,1024)"
    assert feature_map.shape == (7,7,1024),  "feature map does not have shape (7,7,1024)"
    assert flow.shape ==(7,7,2),             "flow does not have shape (7,7,2)" 
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


# speed up by doing flow all at once 
def ASFS(args,rgb_frames,S,fsk):
    print("ASFS")
    tau2 = 0.75
    tau3 = 0.04

    rgb_start_index = 0
    remove_indexes = [] 
    similarties = [] 

    for j,Si in enumerate(S):
        remove_indexes.append([])
        similarties.append([]) 
        #get the refrence rgb_frame and feature map 
        key_frame = fsk[j]
        refrence_frame = rgb_frames[rgb_start_index]
        flow_images = [] 

        for i in range(1,len(Si)):
            frame_index = rgb_start_index + i
            frame = rgb_frames[frame_index] 
            flow_set = [refrence_frame,frame]
            flow_images.extend(flow_set)
        
        flow = flow_from_frames(args,flow_images)
        warped = [] 
        for i,feature_map in enumerate(Si):
            flow_image = flow[i].transpose(1,2,0)
            flow_image = cv2.resize(flow_image,(7,7))
            warped.append(warp(key_frame,flow_image,feature_map))
            sim = SM(i,j,warped,key_frame)
            
            if sim > tau2:
                print(f"TOO SIMILAR FRAME {i}")
                #remove local feature fsi in next step
                remove_indexes[j].append(i)
            elif sim < tau3:
                print("NOISY FRAME")
                # the frame is noisy, discard the deep feature now 
                Si.pop(i+1)
                remove_indexes[j].append(i)
            else:
                similarties[j].append(sim)
    print(remove_indexes)
    for j_ in range(j):
        if remove_indexes[j_]:
            remove_indexes[j_].reverse()
            for i in remove_indexes[j_]:
                print(i)
                S[j_].pop(i)

    # Pool the weight for each segment
    fvsk = [] 
    for j,Si in enumerate(S):
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



