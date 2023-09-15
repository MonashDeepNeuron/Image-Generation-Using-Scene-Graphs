# Visual Genome Dataset Preparation
# STEPS WE NEED TO FOLLOW:
# Extract all objects of an image
# Class ID, X and Y position of the objects in scale [0,1]



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
  object_json_dir

  Inputs:
  - image_folder_dir: Path to a directory where images are held (str-path)
  - instances_json: Path to a JSON file giving COCO annotations (str-path)
  - image_size: Size (H, W) at which to load images. 
  - mask_size: Size M for object segmentation masks; default 16. (int)
  - object_json_dir
  """

  def __init__(self, image_folder_dir, image_json_dir, object_json_dir, image_size=(128, 128),min_object_size=0.02,
               min_objects_per_image=3, max_objects_per_image=8, transform=None, pre_transform=None, pre_filter=None):
   
    # Listing 
    self.image_folder_dir = image_folder_dir
    self.image_size = image_size
    self.image_json_dir = image_json_dir 
    self.object_json_dir = object_json_dir 
    self.min_objects_per_image = min_object_per_imag

  #Open image_json_dir json file 
    with open(image_folder_dir, 'r') as f: # r => the file will be opened in read mode but not for modifying.
      image_json_dir = json.load(f) 

     #Open object_data json file 
    with open(object_json_dir, 'r') as f: # r => the file will be opened in read mode but not for modifying.
      object_data = json.load(f) 


    # image_json_dir Storage
    self.img_id_store = [] 
    self.img_id_to_url= {} 
    self.img_id_to_size = {}
    self.img_id_to_objects = {}

    '''
    Process each image
    '''
    for item in image_json_dir:# IF AN ERROR OCCURS (CANNOT INDEX), for item in image_json_dir['images']:
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
    all_categories = [] ### List of all class names in dataset ###
    obj_id_to_img_id = {} ### maps every object ID to its image ID
    obj_name_to_idx = {}
    obj_idx_to_name ={}
    # extract the categories/names of the object from object_data
    
    
    '''
    Process each object
    '''

    for index in range(0,len(object_data)): # no need len(object_data) - 1 as range(0,x) is not x inclusive
       # object_data[index]: # IF AN ERROR OCCURS (CANNOT INDEX), for item in image_json_dir['categories']:

       # Retrieving the 'Object ID' (e.g. 10000223, 231, the unique numerical identifier for the object)
       object_id = object_data[index]['object_id'] 

       # Retrieving the 'Category Name' of the object (e.g. 'Apple', 'Car')
       category_id = object_data[index]['name']
       
       #object_data[index]['synsets'][0]

      # Mapping the 'Object ID' to its corresponding 'Category Name' 
       obj_idx_to_name[index] = category_id
       obj_name_to_idx[category_id] = index

      # Mapping the 'Object ID' to its corresponding image 'Image ID'
       obj_id_to_img_id[object_id] = object_data[index]['image_id']

      # # Mapping the 'Object ID' to its corresponding image 'Image ID'
      #  self.dict_vocab['object_name_to_idx'][category_name] = category_id
       
       # Building the all_categories list which holds every category this dataset contains (e.g. 'Apple', 'Car')
       if category_id not in all_categories:
          all_categories.append(all_categories)
          

    # Add object data from instances
    self.img_id_to_objects = defaultdict(list) ### This maps an image to its objects, previously we just mapped class id to objects ###

    '''
    Calculating the bounding boxes for each object

    TODO: Calculate x,y coordinates
    '''
    for index in range(0,len(object_data)):
    #for object_data in image_json_dir:
      image_id = object_data[index]['image_id']

      # width and height of the object
      w = object_data[index]['w']
      h = object_data[index]['h']
      x = object_data['x']
      y = object_data['y']
      
      
      # object_data[index]['synsets'][0] = category_id in docodatase

     # width and height of the image
      width,height = self.img_id_to_size[image_id]
      box_area = (w * h) / (width * height) ### used to calculate min box_area
      
      #x0, y0: bottom left
      #x1, y1: top right

      H, W = self.image_size
      objs, boxes = [], []
      for object_data in self.img_id_to_objects[image_id]:
        #  objs.append(object_data['category_id'])
        #  x, y, w, h = object_data['bbox']
          x0 = x / width
          y0 = y / height
          x1 = (x + w) / width
          y1 = (y + h) / height
          boxes.append(torch.FloatTensor([x0, y0, x1, y1]))

          # # This will give a numpy array of shape (HH, WW)
          # mask = self.seg_to_mask(object_data['segmentation'], WW, HH)

          # # Crop the mask according to the bounding box, being careful to
          # # ensure that we don't crop a zero-area region
          # mx0, mx1 = int(round(x)), int(round(x + w))
          # my0, my1 = int(round(y)), int(round(y + h))
          # mx1 = max(mx0 + 1, mx1)
          # my1 = max(my0 + 1, my1)
          # mask = mask[my0:my1, mx0:mx1]
          # mask = imresize(255.0 * mask, (self.mask_size, self.mask_size),
          #                 mode='constant')
          # mask = torch.from_numpy((mask > 128).astype(np.int64))
          # masks.append(mask)
      
      box_ok = box_area > min_object_size
      object_name = object_idx_to_name[object_data[index]['name']] ### Again using dictionaries helpful for referencing
      category_ok = object_name in all_categories
      #other_ok = object_name != 'other' or include_other ### Don't include object if its annotated as "other"
      if box_ok and category_ok:
        self.img_id_to_objects[image_id].append(object_data[index]) ### We can use this object ###
      
      

      # COCO category labels start at 1, so use 0 for __image__
      # self.vocab['object_name_to_idx']['__image__'] = 0 ### supernode ###
      
      # Build object_idx_to_name
      name_to_idx = self.dict_vocab['object_name_to_idx'][category_id]  # we haven't built this yet (errror)

      # self.dict_vocab['object_name_to_idx'] = # cat: 4    self.dict_vocab['object_name_to_idx'] = # cat: 4

'''
self.dict_vocab = {
      'object_name_to_idx': {cat: 4, dog:5}, # 'image_id' to index in img_id_store list 
      'pred_name_to_idx': {}, # pred_name to index in img_id_store list 
  }
'''



      assert len(name_to_idx) == len(set(name_to_idx.values())) ### Assert that the number of classes you have seen is the same as the number of indexes/classes
      max_object_idx = max(name_to_idx.values()) 
      idx_to_name = ['NONE'] * (1 + max_object_idx) ### preallocate dictionary with entries for each class (+1 for supernode)
      for name, idx in self.dict_vocab['object_name_to_idx'].items(): ### Fill out dictionary with 'NONE' class names, which will be replaced with actual class name
          idx_to_name[idx] = name
      self.dict_vocab['object_idx_to_name'] = idx_to_name

      # Prune images that have too few or too many objects
      new_image_ids = []
      total_objs = 0
      for image_id in self.img_id_store: # this will error, hsven't built image_ids
          num_objs = len(self.img_id_to_objects[image_id])
          total_objs += num_objs

          # If t
          if min_objects_per_image <= num_objs <= max_objects_per_image: # min = 3 , max = 8
              new_image_ids.append(image_id)
      self.img_id_store = new_image_ids
      
      self.dict_vocab['pred_idx_to_name'] = [
          '__in_image__', ## for da supernode
          'left of',
          'right of',
          'above',
          'below',
          'inside',
          'surrounding',
      ]
      self.dict_vocab['pred_name_to_idx'] = {}
      for idx, name in enumerate(self.dict_vocab['pred_idx_to_name']):
          self.dict_vocab['pred_name_to_idx'][name] = idx
          
        super().__init__(image_folder_dir, transform, pre_transform, pre_filter)
      
        # Testing
        @property
        def raw_file_names(self):
            # I dont seem to have any validation dataset, need to ask Nyan
            # return ['data/COCO_Stuff/validation/val2017/000000002592.jpg']

        @property
        def processed_file_names(self):
            # return ['data_1.pt']
        
        
        
        '''
         def process(self):
        index = 0
        for raw_path in self.raw_paths:

            image_id = self.image_ids[index]
        
            filename = self.image_id_to_filename[image_id]
            print(filename)
            image_path = os.path.join(self.image_folder_dir, filename)

            with open(image_path, 'rb') as f:
                with PIL.Image.open(f) as image:
                    WW, HH = image.size
                    try:
                        image = self.transform(image.convert('RGB'))
                    except:
                        print("No transform specified")
      
        '''
       
            
      
      

        





       
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
    image_folder_dir = "/Users/abbas/Desktop/Engineering/Coding/Deep-Neuron/Datasets/Image-Generation-Dataset-(Visual-Genome)/Visual_Genome_Part2_Images"
    image_paths = [...]  # List of image file paths
    object_info = [...]  # List of dictionaries containing object info

    image_size = (256, 256)  # Example image size
    dataset = VisualGenomeDataset(image_folder_dir, image_paths, object_info, image_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Now you can use the dataloader to iterate over your dataset in batches
