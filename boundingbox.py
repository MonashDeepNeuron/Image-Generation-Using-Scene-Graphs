

import torch

import torch.nn as nn

# Bringing in modules:
import os
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
#import h5py 
# not sure what this module does and if we need it ^^
import PIL 
from PIL import Image

'''
multilayer perceptrones

1. import all necessary libraries for loading data
2. Define and initialise the neural network (def __init__(self,dim = [dim_in, dim_hidden, dim_out]))
3. Specify how data will pass thr the model (def forward(self, x, y):)
4. pass data thr mode to test (if __name__ == '__main__')




#left_col, top_row, right_col, bottom_row

# init(self, dim=[in, hidden, out])
# in = N
# hidden = 512 (can change this to ur liking)
# out = 4 (x1, x2, y1, y2)
# fc1 = nn.Linear(in, hidden)
# fc2 = nn.Linear(hidden, out)
# relu = nn.Relu()

# forward(self, x):
# x = fc1(x)
# x = relu(x)
# out = fc2(x)
# return out

'''


class boundingboxPredict(nn.Module):
    def __init__(self,dim):
        super(boundingboxPredict, self).__init__()
        
      
      
        # 1st => fully connected layer
        self.fc1 = nn.Linear(dim[0], dim[1])

        #2nd =? fully connected layer that outputs [dim_out] values
        self.fc2 = nn.Linear(dim[1],dim[2])
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)
        return out


    '''
     # pseudo
        self.vect_in = vect_in
        self.obj_vects = obj_vects
        self.left_col = left_col
        self.top_row = top_row
        self.right_col = right_col
        self.bottom_row = bottow_row 
    
    def create_mask_bb(bb,x):
        bb = bb.astype(np,int)


    def extract_bb(obj_vects):
        top_row = np.min(rows)
        left_col = np.min(cols)
        bottom_row = np.max(rows)
        right_col = np.max(cols)
        
        return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)


    def create_bb_array(x):
    """Generates bounding box array from a train_df row"""
        return np.array([x[5],x[4],x[7],x[6]])
    
    
    '''



if __name__ == '__main__':
   
   #define input dimension
    dim_in = 4
    dim_hidden = 512
    dim_out = 4

    instance_bbox = boundingboxPredict(dim=[dim_in, dim_hidden, dim_out])
    
    org_vector = torch.randn(1,dim_in)

    bounding_box = instance_bbox(org_vector)

    print("predicted bbox: ")

    print(bounding_box)






   # ew_feature_vector = torch.tensor([x, y, h, w, category_id, colour_id, object_id], dtype=torch.float)