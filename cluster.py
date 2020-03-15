from numpy import dot
from numpy.linalg import norm
from scipy import spatial
import numpy as np
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

# Adaptive Video Feature Segmentation
def AVFS(f):
    tau = 0.85
    Si = 0

    # Step 1 Construct ActionS 
    S = []
    current_S = []
    current_S.append(f[0])
    
    # Determine if features maps from f2 to fi belong to fsk1 (S[1])
    for i in range(1,len(f)):
        fi = np.squeeze(f[i])
        print(fi.shape)
        print(current_S[0][0].shape)



        sim = CS(current_S[0][0],fi)
        print(sim)
        # end of the current action 
        if sim >= tau:
            print("NEW ACTION")
            S.append(current_S)
            Si+=1
            # reset actions 
            current_S = []
            #new feature index     
        
        current_S.append(fi)


    S.append(current_S)
    print(len(S))


    # use the first frame of the video as the first action
    # S1 = f1 = fsk1


    # Step ? Construct the Codebook C?
    C = []
