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



class VisualGenomeDataset(Dataset):
  """ 
  A PyTorch Dataset for loading Visual Genome dataset image and
  getting information from them.
  

  Inputs:
  - image_dir: Path to a directory where images are held (str-path)
  - instances_json: Path to a JSON file giving COCO annotations (str-path)
  - image_size: Size (H, W) at which to load images. 
  - mask_size: Size M for object segmentation masks; default 16. (int)
  - object_dir
  """

  def __init__(self, image_id, image_dir, image_data,object_dir,object_name, object_id, image_size=(256, 256),min_object_size=0.02):
    super(VisualGenomeDataset, self).__init__() 

    # Listing 
    self.image_dir = image_dir
    self.image_size = image_size
    self.image_id = image_id
    self.object_name = object_name
    self.object_id = object_id
    self.image_data = image_data
    self.object_dir = object_dir 
    

  #Open image_data json file 
    with open(image_dir, 'r') as f: # r => the file will be opened in read mode but not for modifying.
      image_data = json.load(f) 

     #Open object_data json file 
    with open(object_dir, 'r') as f: # r => the file will be opened in read mode but not for modifying.
      object_data = json.load(f) 


    # Image_Data Storage
    self.img_id_store = [] 
    self.img_id_to_url= {} 
    self.img_id_to_size = {}
    self.img_id_to_objects = {}

      
    '''
    Process each image
    '''
    for item in image_data:# IF AN ERROR OCCURS (CANNOT INDEX), for item in image_data['images']:
          image_id =  item['image_id']
          file_url= item['url']
          width = item['width']
          height = item['height']
          self.img_id_store.append(image_id)
          self.img_id_to_url[image_id] = file_url
          self.img_id_to_size[image_id] = (width, height)  #Ok upto here

          
    # Object_Data storage
    # dictionary storing index and name/categories of the image
    self.dict_vocab = {
       'object_name_to_idx': {}, # 'image_id' to index in img_id_store list 
       'pred_name_to_idx': {}, # pred_name to index in img_id_store list 
    }

    object_idx_to_name = {} ### maps every index value to an object name ###
    all_instance_categories = [] ### List of all class names in dataset ###
    obj_id_to_img_id = {} ### maps every object ID to its image ID

    obj_idx_to_name ={}
    # extract the categories/names of the object from object_data
    
    
    '''
    Process each object
    '''

    for index in range(0,len(object_data)): # no need len(object_data) - 1 as range(0,x) is not x inclusive
       # object_data[index]: # IF AN ERROR OCCURS (CANNOT INDEX), for item in image_data['categories']:

       # Retrieving the 'Object ID' (e.g. 10000223, 231, the unique numerical identifier for the object)
       object_id = object_data[index]['object_id'] 

       # Retrieving the 'Category Name' of the object (e.g. 'Apple', 'Car')
       category_id = object_data[index]['synsets'][0]
       
       #object_data[index]['synsets'][0]

      # Mapping the 'Object ID' to its corresponding 'Category Name' 
       obj_idx_to_name[object_id] = category_id

      # Mapping the 'Object ID' to its corresponding image 'Image ID'
       obj_id_to_img_id[object_id] = object_data[index]['image_id']

      # # Mapping the 'Object ID' to its corresponding image 'Image ID'
      #  self.dict_vocab['object_name_to_idx'][category_name] = category_id
       
       # Building the all_instance_categories list which holds every category this dataset contains (e.g. 'Apple', 'Car')
       if category_id not in all_instance_categories:
          all_instance_categories.append(category_id)
          

    # Add object data from instances
    self.img_id_to_objects = defaultdict(list) ### This maps an image to its objects, previously we just mapped class id to objects ###

    '''
    Calculating the bounding boxes for each object
    '''
    for index in range(0,len(object_data)):
    #for object_data in image_data:
      image_id = object_data[index]['image_id']

      # width and height of the object
      w = object_data[index]['w']
      h = object_data[index]['h']
      
      # object_data[index]['synsets'][0] = category_id in docodatase


     # width and height of the image
      width,height = self.img_id_to_size[image_id]
      box_area = (w * h) / (width * height) ### used to calculate min box_area
      box_ok = box_area > min_object_size
      object_name = object_idx_to_name[object_data[index]['synsets'][0]] ### Again using dictionaries helpful for referencing
      category_ok = object_name in category_whitelist 
      other_ok = object_name != 'other' or include_other ### Don't include object if its annotated as "other"
      if box_ok and category_ok and other_ok:
        self.image_id_to_objects[image_id].append(object_data[index]) ### We can use this object ###







       
  '''
  if stuff_data:
      for category_data in stuff_data['categories']:
          category_name = category_data['name']
          category_id = category_data['id']
          all_stuff_categories.append(category_name) ### Stores reference of the current "stuff" name ###
          object_idx_to_name[category_id] = category_name ### Treats "stuff" and object instances as the same, useful for referencing ###
          self.vocab['object_name_to_idx'][category_name] = category_id
  '''


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
       
      
  
    self.sceneGraphConstructor = mst_algo.SceneGraphConstructor(vocab=self.vocab, boxes=boxes)
    scene_graph = self.sceneGraphConstructor.construct_scene_graph(objs, boxes, masks)
    g = torch_geometric.utils.to_networkx(scene_graph, to_undirected=False)
    nx.draw(g)
    torch.save(scene_graph, "please_god.pt")


# Main Function:


if __name__ == "__main__":
    image_dir = "/Users/abbas/Desktop/Engineering/Coding/Deep-Neuron/Datasets/Image-Generation-Dataset-(Visual-Genome)/Visual_Genome_Part2_Images"
    image_paths = [...]  # List of image file paths
    object_info = [...]  # List of dictionaries containing object info

    image_size = (256, 256)  # Example image size
    dataset = VisualGenomeDataset(image_dir, image_paths, object_info, image_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Now you can use the dataloader to iterate over your dataset in batches
