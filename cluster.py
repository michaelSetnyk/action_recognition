from numpy import dot
from numpy.linalg import norm
from scipy import spatial
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree


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
    k = 16
    tau = 0.85
    f = [fi.flatten() for fi in f]
    # Step 1 construct the Codebook C
    
    C = KMeans(n_clusters=k,init='k-means++',tol=0.0001).fit(f)
    
    # Step 2 Construct ActionS 
    S = []

    # set of indexes of key frames 
    fsk = [0]

    Si = [f[0]]
    fski = f[0]
    # Determine if features maps from f2 to fi belong to fsk1 (S[1])
    for i in range(1,len(f)):
        fi = f[i]
        sim = CS(fski,fi)
        # end of the current action 
        if sim < tau:
            print("NEW ACTION")
            S.append(Si)
            Si = []
            fski = fi
            fsk.append(i)
        Si.append(fi)
    S.append(Si)
    
    [print(f"S[{i}] = {len(S[i])}") for i in range(len(S))]
    # Step 3 segmentation updating 
    for s_i, i in enumerate(fsk):
        prev = f[i-1]
        fi = f[i]

        prev_label =  C.labels_[i-1]
        label =  C.labels_[i]
       
        # Update labels
        if (label == prev_label and s_i != len(S)):
            # update fi to fi -1 
            S[s_i].append(fi)
            # remove fi from Si
            s_a=s_i + 1
            
            if s_a >= len(S):
                break
            S[s_a].pop(0)
            
            # update neighbours fi + a (a is up to k)
            a=i + 1
            neighbors_similar = True 
            while neighbors_similar: 
                next_label = C.labels_[a]
                if not S[s_a]:
                    fsk.pop(s_a)
                    S.pop(s_a)
                    if s_a >= len(S):
                        break
                if label == next_label:
                    S[s_i].append(f[a])
                    S[s_a].pop(0)
                else:
                    neighbors_similar = False
                a += 1
    [print(f"S[{i}] = {len(S[i])}") for i in range(len(S))]
    # Maybe update the fsk to make sure the feature is in the set 


    print(fsk)

'''
Adaptive Segment Feature Sampling
'''
def ASFS(S,fsk):
    for j in range(len(S)):
        Si = S[j]
        fski = fsk[j]
        for i in range(len(Si)):
            pass


    





