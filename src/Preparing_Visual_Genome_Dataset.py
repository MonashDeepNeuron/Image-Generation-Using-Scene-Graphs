# Preparing Visual Genome Dataset (idk what we doing but we ball)

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

#from .utils import imagenet_preprocess, Resize


## Figure out how to open json files 

# Firstly, trying to make a class to store all the variables that we need 
class VisualGenomeDataset(Dataset):
  # class attributes:
  # (Placeholder)


  # Constructor
  def __init__(self, image_id, object_name, object_id, x0, y0, x1, y1, image_size): 
    # Defining the variables
    self.image_id = image_id
    self.object_name = object_name
    self.object_id = object_id
    self.border_info = border_info
    self.image_size = image_size
    
    self.transform = T.Compose([T.Resize(image_size), T.ToTensor()])

    border_info ={y0_bottom_left:0, y1_top_left:0, x0_bottom_left:0, y1_top_right:0, x_width:0 , y_height:0}
    
    # pass all info into dictionary border_info
    
    border_info[x0_bottom_left]= x0
    border_info[y0_bottom_left]=  y0
    border_info[x1_top_right]= x1
    border_info[y1_top_right]= y1
    border_info[x_width]= img_dimension[0]
    border_info[y_height] = img_dimension[1]

    # [x_width, y_height]
    self.img_dimension = img_dimension
    img_dimension = []
    #img_dimension[0]= self

  # Psuedo
    bt_left_x0y0 = [x0,y0] 
    tp_right_x1y1 = [x1,y1]

    # check 
    resize = (0.5,0.5) # Between 0 and 1 (worry about later)
    org_size = [bt_left_x0y0, tp_right_x1y1]
    org_size[0] = x0_bottom_left





  # Trying to extract image dimensions from an image (leave for later)
  def get_image_size(self,index):
    index = 0
    img_path = os.path.join(self.image_dir, self.image_paths[index])
  
  def __len__(self):
        return len(self.image_paths)

   
  def __getitem__(self, index):
    img_path = os.path.join(self.image_dir, self.image_paths[index])
        
      with open(img_path, 'rb') as f:
          with PIL.Image.open(f) as image:
              width, height = image.size
              image = self.transform(image.convert('RGB'))
        
      # Process the object information here
      object_info = self.object_info[index]

      return {
            'image': image,
            'object_info': object_info,
            'image_size': (width, height)
        }

# Main Function:


if __name__ == "__main__":
    image_directory = "/Users/abbas/Desktop/Engineering/Coding/Deep-Neuron/Datasets/Image-Generation-Dataset-(Visual-Genome)/Visual_Genome_Part2_Images"
    image_paths = [...]  # List of image file paths
    object_info = [...]  # List of dictionaries containing object info

    image_size = (256, 256)  # Example image size

    dataset = VisualGenomeDataset(image_directory, image_paths, object_info, image_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Now you can use the dataloader to iterate over your dataset in batches