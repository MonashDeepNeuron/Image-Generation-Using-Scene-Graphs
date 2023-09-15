import json
import os
import random
import math
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import PIL
from skimage.transform import resize as imresize
import pycocotools.mask as mask_utils

# from .utils import imagenet_preprocess, Resize
import torch_geometric
from torch_geometric.data import Dataset, download_url
import networkx as nx
from networkx.algorithms import tree

import mst_algo
import delaunay


class Resize(object):
    def __init__(self, size, interp=PIL.Image.BILINEAR):
        if isinstance(size, tuple):
            H, W = size
            self.size = (W, H)
        else:
            self.size = (size, size)
        self.interp = interp

    def __call__(self, img):
        return img.resize(self.size, self.interp)


class CocoPyGDataset(Dataset):
    """
    A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
    them to scene graphs on the fly.

    Inputs:
    - image_dir: Path to a directory where images are held (str-path)
    - instances_json: Path to a JSON file giving COCO annotations (str-path)
    - stuff_json: (optional) Path to a JSON file giving COCO-Stuff annotations (str-path)
    - stuff_only: (optional, default True) If True then only iterate over (boolean)
      images which appear in stuff_json; if False then iterate over all images
      in instances_json. 
    - image_size: Size (H, W) at which to load images. Default (64, 64). (int,int)
    - mask_size: Size M for object segmentation masks; default 16. (int)
    - normalize_image: If True then norm alize images by subtracting ImageNet
      mean pixel and dividing by ImageNet std pixel. (boolean)
    - max_samples: If None use all images. Other wise only use images in the
      range [0, max_samples). Default None. (int)
    - include_relationships: If True then include spatial relationships; if
      False then only include the trivial __in_image__ relationship. (boolean)
    - min_object_size: Ignore objects whose bounding box takes up less than
      this fraction of the image. (int)
    - min_objects_per_image: Ignore images which have fewer than this many
      object annotations. (int)
    - max_objects_per_image: Ignore images which have more than this many
      object annotations. (int)
    - include_other: If True, include COCO-Stuff annotations which have category
      "other". Default is False, because I found that these were really noisy (boolean)
      and pretty much impossible for the system to model.
    - instance_whitelist: None means use all instance categories. Otherwise a (list)
      list giving a whitelist of instance category names to use.
    - stuff_whitelist: None means use all stuff categories. Otherwise a list (list)
      giving a whitelist of stuff category names to use.
    """

    def __init__(self, image_dir, instances_json, stuff_json=None,
                 stuff_only=True, image_size=(64, 64), mask_size=16,
                 normalize_images=True, max_samples=None,
                 include_relationships=True, min_object_size=0.02,
                 min_objects_per_image=3, max_objects_per_image=8,
                 include_other=False, instance_whitelist=None, stuff_whitelist=None, transform=None, pre_transform=None, pre_filter=None):

        if stuff_only and stuff_json is None:
            print('WARNING: Got stuff_only=True but stuff_json=None.')
            print('Falling back to stuff_only=False.')

        self.image_dir = image_dir  # str-path
        self.mask_size = mask_size  # int  ### COCO-Stuff pixel segmentation mask size
        self.max_samples = max_samples  # boolean
        self.normalize_images = normalize_images  # boolean
        self.include_relationships = include_relationships
        self.set_image_size(image_size)  # For resizing image

        with open(instances_json, 'r') as f:
            instances_data = json.load(f)  # Loading COCO annotations

        # access data, point to the right object
        stuff_data = None
        if stuff_json is not None and stuff_json != '':
            with open(stuff_json, 'r') as f:
                stuff_data = json.load(f)  # Loading COCO-Stuff annotations

        ### Storing COCO image ids and filename/size ###
        self.image_ids = []
        # Dictionaries to store references between image and filename
        self.image_id_to_filename = {}
        # Can be used to get the width/height of the image using the id of the image
        self.image_id_to_size = {}
        for image_data in instances_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']
            width = image_data['width']
            height = image_data['height']
            self.image_ids.append(image_id)  # self.image_ids = [id1, id2,...]
            # self.image_id_to_filename = {id1:file1, id2: file2:....}
            self.image_id_to_filename[image_id] = filename
            # self.image_id_to_size = {id1:(w,h), id2:(w,h),...}
            self.image_id_to_size[image_id] = (width, height)

        ### This dictionary stores two subdictionaries ###
        ### object_name_to_idx - maps every type of class/object name to an index value ###
        ### pred_name_to_idx - maps every predicate name to an index value ###
        ### These dictionaries allow us to easily construct the scene graphs ###
        self.vocab = {
            'object_name_to_idx': {},
            'pred_name_to_idx': {},
        }
        object_idx_to_name = {}  # maps every index value to an object name ###
        all_instance_categories = []  # List of all class names in dataset ###

        for category_data in instances_data['categories']:
            category_id = category_data['id']  # Get class ID ###
            category_name = category_data['name']  # Get class name ###
            # Stores reference of the current object name ###
            all_instance_categories.append(category_name)
            #  all_instance_categories = ['name1','name2',...]
            # Update dictionary if new class ID encountered ###
            object_idx_to_name[category_id] = category_name
            # object_idx_to_name = {ctid1: ctname1, ctid2: ctname2,...}
            # Update dictionary if new class name encountered, make sure mapping is consistent ###
            self.vocab['object_name_to_idx'][category_name] = category_id
            # check vice versa
        all_stuff_categories = []  # List of all stuff names in dataset ###

        if stuff_data:
            for category_data in stuff_data['categories']:
                category_name = category_data['name']
                category_id = category_data['id']
                # Stores reference of the current "stuff" name ###
                all_stuff_categories.append(category_name)
                # Treats "stuff" and object instances as the same, useful for referencing ###
                object_idx_to_name[category_id] = category_name
                self.vocab['object_name_to_idx'][category_name] = category_id

        if instance_whitelist is None:
            instance_whitelist = all_instance_categories
        if stuff_whitelist is None:
            stuff_whitelist = all_stuff_categories
        category_whitelist = set(instance_whitelist) | set(
            stuff_whitelist)  # Only include the whitelisted categories ###

        # Add object data from instances
        # This maps an image to its objects, previously we just mapped class id to objects ###
        self.image_id_to_objects = defaultdict(list)

        # disregard the object if its border (area) not satisfy

        for object_data in instances_data['annotations']:
            image_id = object_data['image_id']
            _, _, w, h = object_data['bbox']
            W, H = self.image_id_to_size[image_id]
            box_area = (w * h) / (W * H)  # used to calculate min box_area
            box_ok = box_area > min_object_size
            # Again using dictionaries helpful for referencing
            object_name = object_idx_to_name[object_data['category_id']]
            category_ok = object_name in category_whitelist
            # Don't include object if its annotated as "other"
            other_ok = object_name != 'other' or include_other
            if box_ok and category_ok and other_ok:
                self.image_id_to_objects[image_id].append(
                    object_data)  # We can use this object ###

        # Add object data from stuff
        if stuff_data:
            # if an image contains stuff annotations, its a subset of all the images ###
            image_ids_with_stuff = set()
            for object_data in stuff_data['annotations']:
                image_id = object_data['image_id']
                image_ids_with_stuff.add(image_id)
                _, _, w, h = object_data['bbox']
                W, H = self.image_id_to_size[image_id]
                box_area = (w * h) / (W * H)
                box_ok = box_area > min_object_size
                object_name = object_idx_to_name[object_data['category_id']]
                category_ok = object_name in category_whitelist
                other_ok = object_name != 'other' or include_other
                if box_ok and category_ok and other_ok:
                    self.image_id_to_objects[image_id].append(
                        object_data)  # Very similar process to above
            if stuff_only:
                new_image_ids = []
                for image_id in self.image_ids:
                    if image_id in image_ids_with_stuff:
                        # only use images with stuff in them
                        new_image_ids.append(image_id)
                self.image_ids = new_image_ids

                all_image_ids = set(self.image_id_to_filename.keys())
                # i'm guessing this is to avoid double ups?  not quite sure yet
                image_ids_to_remove = all_image_ids - image_ids_with_stuff
                for image_id in image_ids_to_remove:
                    self.image_id_to_filename.pop(image_id, None)
                    self.image_id_to_size.pop(image_id, None)
                    self.image_id_to_objects.pop(image_id, None)

        # COCO category labels start at 1, so use 0 for __image__
        self.vocab['object_name_to_idx']['__image__'] = 0  # supernode ###

        # Build object_idx_to_name
        name_to_idx = self.vocab['object_name_to_idx']
        # Assert that the number of classes you have seen is the same as the number of indexes/classes
        assert len(name_to_idx) == len(set(name_to_idx.values()))
        max_object_idx = max(name_to_idx.values())
        # preallocate dictionary with entries for each class (+1 for supernode)
        idx_to_name = ['NONE'] * (1 + max_object_idx)
        # Fill out dictionary with 'NONE' class names, which will be replaced with actual class name
        for name, idx in self.vocab['object_name_to_idx'].items():
            idx_to_name[idx] = name
        self.vocab['object_idx_to_name'] = idx_to_name

        # Prune images that have too few or too many objects
        new_image_ids = []
        total_objs = 0
        for image_id in self.image_ids:
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
            if min_objects_per_image <= num_objs <= max_objects_per_image:
                new_image_ids.append(image_id)
        self.image_ids = new_image_ids

        self.vocab['pred_idx_to_name'] = [
            '__in_image__',  # for da supernode
            'left of',
            'right of',
            'above',
            'below',
            'inside',
            'surrounding',
        ]
        self.vocab['pred_name_to_idx'] = {}
        for idx, name in enumerate(self.vocab['pred_idx_to_name']):
            self.vocab['pred_name_to_idx'][name] = idx

        super().__init__(image_dir, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['data/COCO_Stuff/validation/val2017/000000002592.jpg']

    @property
    def processed_file_names(self):
        return ['data_1.pt']

    # def download(self):
    #     # Download to `self.raw_dir`.
    #     path = download_url(url, self.raw_dir)
    #     ...

    def process(self):
        index = 0
        for raw_path in self.raw_paths:

            image_id = self.image_ids[index]

            filename = self.image_id_to_filename[image_id]
            print(filename)
            image_path = os.path.join(self.image_dir, filename)

            with open(image_path, 'rb') as f:
                with PIL.Image.open(f) as image:
                    WW, HH = image.size
                    try:
                        image = self.transform(image.convert('RGB'))
                    except:
                        print("No transform specified")

            H, W = self.image_size
            objs, boxes, masks = [], [], []
            for object_data in self.image_id_to_objects[image_id]:
                objs.append(object_data['category_id'])
                x, y, w, h = object_data['bbox']
                x0 = x / WW
                y0 = y / HH
                x1 = (x + w) / WW
                y1 = (y + h) / HH
                boxes.append(torch.FloatTensor([x0, y0, x1, y1]))

                # This will give a numpy array of shape (HH, WW)
                mask = self.seg_to_mask(object_data['segmentation'], WW, HH)

                # Crop the mask according to the bounding box, being careful to
                # ensure that we don't crop a zero-area region
                mx0, mx1 = int(round(x)), int(round(x + w))
                my0, my1 = int(round(y)), int(round(y + h))
                mx1 = max(mx0 + 1, mx1)
                my1 = max(my0 + 1, my1)
                mask = mask[my0:my1, mx0:mx1]
                mask = imresize(255.0 * mask, (self.mask_size, self.mask_size),
                                mode='constant')
                mask = torch.from_numpy((mask > 128).astype(np.int64))
                masks.append(mask)

            # Add dummy __image__ object
            # entire image is the supernode
            objs.append(self.vocab['object_name_to_idx']['__image__'])
            boxes.append(torch.FloatTensor([0, 0, 1, 1]))
            masks.append(torch.ones(self.mask_size, self.mask_size).long())

            objs = torch.LongTensor(objs)
            boxes = torch.stack(boxes, dim=0)
            masks = torch.stack(masks, dim=0)

            self.sceneGraphConstructor = delaunay.SceneGraphConstructor(
                vocab=self.vocab, boxes=boxes)
            scene_graph = self.sceneGraphConstructor.construct_scene_graph(
                objs, boxes, masks)
            g = torch_geometric.utils.to_networkx(
                scene_graph, to_undirected=False)
            nx.draw(g)
            torch.save(scene_graph, "please_god.pt")

    def get_adjacency_info(self, objs, boxes, masks):
        # Directed edges, subject -> predicate -> object
        pass

    def total_objects(self):
        total_objs = 0
        for i, image_id in enumerate(self.image_ids):
            if self.max_samples and i >= self.max_samples:
                break
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
        return total_objs

    def len(self):
        if self.max_samples is None:
            return len(self.image_ids)
        return min(len(self.image_ids), self.max_samples)

    def get(self, idx):
        pass
    #   data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
    #   return data

    def seg_to_mask(self, seg, width=1.0, height=1.0):
        """
        Tiny utility for decoding segmentation masks using the pycocotools API.
        """
        if type(seg) == list:
            rles = mask_utils.frPyObjects(seg, height, width)
            rle = mask_utils.merge(rles)
        elif type(seg['counts']) == list:
            rle = mask_utils.frPyObjects(seg, height, width)
        else:
            rle = seg
        return mask_utils.decode(rle)

    def set_image_size(self, image_size):
        print('called set_image_size', image_size)
        transform = [Resize(image_size), T.ToTensor()]
        # if self.normalize_images:
        #     transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)
        self.image_size = image_size


if __name__ == "__main__":
    image_dir = "/Users/nyankyaw/Documents/UNI/YEAR4/MDN/Image-Generation-Using-Scene-Graphs/data/COCO_Stuff/validation/val2017"
    instances_json = "/Users/nyankyaw/Documents/UNI/YEAR4/MDN/Image-Generation-Using-Scene-Graphs/data/COCO_Stuff/thing_annotations_coco/annotations/instances_val2017.json"
    stuff_json = "/Users/nyankyaw/Documents/UNI/YEAR4/MDN/Image-Generation-Using-Scene-Graphs/data/COCO_Stuff/stuff_annotations_coco/stuff_val2017.json"
    l = {i: i + 1 for i in range(9)}

    dset = CocoPyGDataset(image_dir, instances_json, stuff_json=stuff_json,
                          stuff_only=True, image_size=(64, 64), mask_size=16,
                          normalize_images=False, max_samples=None,
                          include_relationships=True, min_object_size=0.02,
                          min_objects_per_image=3, max_objects_per_image=8,
                          include_other=False, instance_whitelist=None, stuff_whitelist=None, transform=None, pre_transform=None, pre_filter=None)
