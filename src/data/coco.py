import json, os, random, math
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import PIL
from skimage.transform import resize as imresize
import pycocotools.mask as mask_utils

# from .utils import imagenet_preprocess, Resize

from torch_geometric.data import Dataset, download_url

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

        self.image_dir = image_dir #str-path
        self.mask_size = mask_size  #int  ### COCO-Stuff pixel segmentation mask size
        self.max_samples = max_samples #boolean
        self.normalize_images = normalize_images #boolean
        self.include_relationships = include_relationships 
        self.set_image_size(image_size)   ### For resizing image

        with open(instances_json, 'r') as f:
            instances_data = json.load(f)  ### Loading COCO annotations

        # access data, point to the right object 
        stuff_data = None
        if stuff_json is not None and stuff_json != '':
            with open(stuff_json, 'r') as f:
                stuff_data = json.load(f)   ### Loading COCO-Stuff annotations

        ### Storing COCO image ids and filename/size ###
        self.image_ids = []
        self.image_id_to_filename = {}   ### Dictionaries to store references between image and filename
        self.image_id_to_size = {}   ### Can be used to get the width/height of the image using the id of the image
        for image_data in instances_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']
            width = image_data['width']
            height = image_data['height']
            self.image_ids.append(image_id) #self.image_ids = [id1, id2,...]
            self.image_id_to_filename[image_id] = filename  #self.image_id_to_filename = {id1:file1, id2: file2:....} 
            self.image_id_to_size[image_id] = (width, height)  #self.image_id_to_size = {id1:(w,h), id2:(w,h),...} 
            
        ### This dictionary stores two subdictionaries ###
        ### object_name_to_idx - maps every type of class/object name to an index value ###
        ### pred_name_to_idx - maps every predicate name to an index value ###
        ### These dictionaries allow us to easily construct the scene graphs ###
        self.vocab = {
            'object_name_to_idx': {},
            'pred_name_to_idx': {},
        }
        object_idx_to_name = {} ### maps every index value to an object name ###
        all_instance_categories = [] ### List of all class names in dataset ###


        for category_data in instances_data['categories']:
            category_id = category_data['id'] ### Get class ID ###
            category_name = category_data['name'] ### Get class name ###
            all_instance_categories.append(category_name) ### Stores reference of the current object name ###
            #  all_instance_categories = ['name1','name2',...]
            object_idx_to_name[category_id] = category_name ### Update dictionary if new class ID encountered ###
            # object_idx_to_name = {ctid1: ctname1, ctid2: ctname2,...}
            self.vocab['object_name_to_idx'][category_name] = category_id ### Update dictionary if new class name encountered, make sure mapping is consistent ###
                                    # check vice versa
        all_stuff_categories = [] ### List of all stuff names in dataset ###
    
        if stuff_data:
            for category_data in stuff_data['categories']:
                category_name = category_data['name']
                category_id = category_data['id']
                all_stuff_categories.append(category_name) ### Stores reference of the current "stuff" name ###
                object_idx_to_name[category_id] = category_name ### Treats "stuff" and object instances as the same, useful for referencing ###
                self.vocab['object_name_to_idx'][category_name] = category_id

        if instance_whitelist is None:
            instance_whitelist = all_instance_categories
        if stuff_whitelist is None:
            stuff_whitelist = all_stuff_categories
        category_whitelist = set(instance_whitelist) | set(stuff_whitelist) ### Only include the whitelisted categories ###

        # Add object data from instances
        self.image_id_to_objects = defaultdict(list) ### This maps an image to its objects, previously we just mapped class id to objects ###
        
        #disregard the object if its border (area) not satisfy
        
        for object_data in instances_data['annotations']:
            image_id = object_data['image_id']
            _, _, w, h = object_data['bbox']  
            W, H = self.image_id_to_size[image_id]
            box_area = (w * h) / (W * H) ### used to calculate min box_area
            box_ok = box_area > min_object_size
            object_name = object_idx_to_name[object_data['category_id']] ### Again using dictionaries helpful for referencing
            category_ok = object_name in category_whitelist 
            other_ok = object_name != 'other' or include_other ### Don't include object if its annotated as "other"
            if box_ok and category_ok and other_ok:
                self.image_id_to_objects[image_id].append(object_data) ### We can use this object ###

        # Add object data from stuff
        if stuff_data:
            image_ids_with_stuff = set() ### if an image contains stuff annotations, its a subset of all the images ###
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
                    self.image_id_to_objects[image_id].append(object_data) ### Very similar process to above
            if stuff_only:
                new_image_ids = []
                for image_id in self.image_ids:
                    if image_id in image_ids_with_stuff:
                        new_image_ids.append(image_id) ### only use images with stuff in them
                self.image_ids = new_image_ids

                all_image_ids = set(self.image_id_to_filename.keys())
                image_ids_to_remove = all_image_ids - image_ids_with_stuff # i'm guessing this is to avoid double ups?  not quite sure yet
                for image_id in image_ids_to_remove:
                    self.image_id_to_filename.pop(image_id, None)
                    self.image_id_to_size.pop(image_id, None)
                    self.image_id_to_objects.pop(image_id, None)

        # COCO category labels start at 1, so use 0 for __image__
        self.vocab['object_name_to_idx']['__image__'] = 0 ### supernode ###

        # Build object_idx_to_name
        name_to_idx = self.vocab['object_name_to_idx'] 
        assert len(name_to_idx) == len(set(name_to_idx.values())) ### Assert that the number of classes you have seen is the same as the number of indexes/classes
        max_object_idx = max(name_to_idx.values()) 
        idx_to_name = ['NONE'] * (1 + max_object_idx) ### preallocate dictionary with entries for each class (+1 for supernode)
        for name, idx in self.vocab['object_name_to_idx'].items(): ### Fill out dictionary with 'NONE' class names, which will be replaced with actual class name
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
            '__in_image__', ## for da supernode
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
        return ['data/COCO_Stuff/validation/val2017/000000000139.jpg']

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
            # Read data from `raw_path`.
            # data = Data(...)

            # if self.pre_filter is not None and not self.pre_filter(data):
            #     continue

            # if self.pre_transform is not None:
            #     data = self.pre_transform(data)

            # torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            # idx += 1

            image_id = self.image_ids[index]
        
            filename = self.image_id_to_filename[image_id]
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
            objs.append(self.vocab['object_name_to_idx']['__image__']) ## entire image is the supernode
            boxes.append(torch.FloatTensor([0, 0, 1, 1]))
            masks.append(torch.ones(self.mask_size, self.mask_size).long())

            objs = torch.LongTensor(objs)
            boxes = torch.stack(boxes, dim=0)
            masks = torch.stack(masks, dim=0)

            self.construct_node_matrix(objs)
            '''
            objs parameter is a 
            '''
            # box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            # # Compute centers of all objects
            '''
            compute the centre of al objects, with edge case for if the mask is empty: taking the midpoint of the bounding box
            '''
            # obj_centers = []
            # _, MH, MW = masks.size()
            # for i, obj_idx in enumerate(objs):
            #     x0, y0, x1, y1 = boxes[i]
            #     mask = (masks[i] == 1) ### only for objects
            #     xs = torch.linspace(x0, x1, MW).view(1, MW).expand(MH, MW)
            #     ys = torch.linspace(y0, y1, MH).view(MH, 1).expand(MH, MW)
            #     if mask.sum() == 0: ### compute means of objects, if for some reason object mask sums to zero just take midpoint
            #         mean_x = 0.5 * (x0 + x1)
            #         mean_y = 0.5 * (y0 + y1)
            #     else:
            #         mean_x = xs[mask].mean()
            #         mean_y = ys[mask].mean()
            #     obj_centers.append([mean_x, mean_y])
            # obj_centers = torch.FloatTensor(obj_centers)

            '''
            for computational efficiency, select one adjacent object
            '''
            # # Add triples
            # triples = []
            # num_objs = objs.size(0)
            # __image__ = self.vocab['object_name_to_idx']['__image__']
            # real_objs = []
            # if num_objs > 1: # if there are objects in image
            #     real_objs = (objs != __image__).nonzero().squeeze(1) # real_objs is a numpy array/tensor storing only objects
            # for cur in real_objs:
            #     choices = [obj for obj in real_objs if obj != cur] # remove self reflexive objects (objects cite themselves as a choice to be adjacent to)
            #     if len(choices) == 0 or not self.include_relationships:
            #         break
            #     other = random.choice(choices) # choose just one choice to improve the relation of???
            #     if random.random() > 0.5: # something to do with relationship symmetries lol stoupid
            #         s, o = cur, other
            #     else:
            #         s, o = other, cur

            # # Check for inside / surrounding
            # sx0, sy0, sx1, sy1 = boxes[s] ## subject corners
            # ox0, oy0, ox1, oy1 = boxes[o] ## object corners 
            # d = obj_centers[s] - obj_centers[o] # d is a vector yeeeee
            # theta = math.atan2(d[1], d[0])  ## angle lol -> opp and adjacent
            # if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
            #     p = 'surrounding'
            # elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
            #     p = 'inside'
            # elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
            #     p = 'left of'
            # elif -3 * math.pi / 4 <= theta < -math.pi / 4:
            #     p = 'above'
            # elif -math.pi / 4 <= theta < math.pi / 4:
            #     p = 'right of'
            # elif math.pi / 4 <= theta < 3 * math.pi / 4:
            #     p = 'below'
            # p = self.vocab['pred_name_to_idx'][p]
            # triples.append([s, p, o])

            # # Add __in_image__ triples
            # O = objs.size(0)
            # in_image = self.vocab['pred_name_to_idx']['__in_image__']
            # for i in range(O - 1):
            #     triples.append([i, in_image, O - 1])
    
            # triples = torch.LongTensor(triples)

        # return image, objs, boxes, masks, triples

    def construct_node_matrix(self, objs):
        '''
        This method constructs the adjacency matrix and node 
        feature vectors for a given obj (list of nodes in the 
        form [vertex ID, position vector]). 

        1. The centres of all nodes are calculated and stored
        in 'objs_centers'. This simplifies each node to a single 
        coordinate . 

        2. 'Edges' are made between each pair of nodes with weight
        being the Euclidean distance between their centres. These are
        added to a NetworkX graph, along with the list of vertex IDs

        3. Kruskals is run on this graph, returning the list of edges
        'edge_list' with each entry being an edge in the form
         (u,v,dist). These edges make up the minimum spanning tree
         (edge_list is the adjacency list for Data graph, remove the dist info)
        TODO:
        4. Add all nodes to the Data graph (with coordinates and object class ID
        in their node feature vector)

        5. For each u, v in each edge in 'edge_list', construct a new 
        relationship node based on u and v's relative positions, add
        this node to node_feat tensor in graph and add edge to edge_list 
        in graph 

        5. Graph canonlicalisation  
       '''
        print(objs)
        all_node_features = []

        for obj in objs:
            obj_feats = [] # how do we want to design the node feature matrix, what is the important information? If we are having relationships as a node, how does this impact?

            all_node_features.append(obj_feats)
        return all_node_features

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
    dset = CocoPyGDataset(image_dir, instances_json, stuff_json=stuff_json,
               stuff_only=True, image_size=(64, 64), mask_size=16,
               normalize_images=False, max_samples=None,
               include_relationships=True, min_object_size=0.02,
               min_objects_per_image=3, max_objects_per_image=8,
               include_other=False, instance_whitelist=None, stuff_whitelist=None, transform=None, pre_transform=None, pre_filter=None)
    