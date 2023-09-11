# Visual Genome Dataset Preparation
# STEPS WE NEED TO FOLLOW:
# Extract all objects of an image
# Class ID, X and Y position of the objects in scale [0,1]
# Use X and Y position to calculate the relationships
# using kurskal algorthim, construct edges between objects (left of, right of)
# Connecting the nodes of relationships 
#  then connecting to supernodes


# Bringing in modules/dependencies/etc:
import os
import random
import math
import json

from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import h5py
import PIL

import torch_geometric
from torch_geometric.data import Dataset, download_url
import networkx as nx
from networkx.algorithms import tree

from .utils import imagenet_preprocess, Resize



class VisualGenomeDataset(Dataset):
  """ 
  A PyTorch Dataset for loading Visual Genome dataset image and converting
  getting information from them.
  

  Inputs:
  - image_dir: Path to a directory where images are held (str-path)
  - instances_json: Path to a JSON file giving COCO annotations (str-path)
  - image_size: Size (H, W) at which to load images. 
  - mask_size: Size M for object segmentation masks; default 16. (int)
  """

  def __init__(self, image_id, image_dir, object_name, object_id, image_size=(256, 256)):
    super(VisualGenomeDataset, self).__init__()

    self.image_dir = image_dir
    self.image_size = image_size
    self.image_id = image_id
    self.object_name = object_name
    

  #Open file 
    with open(image_data, 'r') as f: # r => the file will be opened in read mode but not for modifying.
      image_data = json.load(f)
    

    with open(object_json, 'r') as f: # r => the file will be opened in read mode but not for modifying.
      object_data = json.load(f)


     # Storage
    self.img_id_store = []
    self.img_id_to_url = {}
    self.obj_id_to_size = {}
    


    for item in object_data['images']:
      object_id =  object_data['image_id']
      file_url= image_data['url']
      width = object_data['w']
      height = object_data['h']
      self.img_id_store.append(object_id)
      self.img_id_to_url[object_id] = file_url
      self.obj_id_to_size[object_id] = (width, height)

    # dictionary storing index and name/categories of the image
    self.dict_vocab = {
       'object_name_to_idx': {},
       'pred_name_to_idx': {},
  
    }

    obj_idx_to_name ={}
    # extract the categories/names of the object from object_data
    all_object_categories= []
    for category_data in object_data:
       category_id = category_data['object_id']
       category_name =  category_data['names']
       all_object_categories.append[category_name]
       obj_idx_to_name[category_id] = category_name
       self.dict_vocab['object_name_to_idx'][category_name] = category_id

    '''
   
    all_stuff_categories = []
      if stuff_data:
        for category_data in stuff_data['categories']:
          category_name = category_data['name']
          category_id = category_data['id']
          all_stuff_categories.append(category_name)
          obj_idx_to_name[category_id] = category_name
          self.vocab['object_name_to_idx'][category_name] = category_id
     '''
       
       


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
