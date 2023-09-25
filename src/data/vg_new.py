# Bringing in modules/dependencies/etc:

import ijson  # for testing
from torch_geometric.data import Dataset, download_url
import torchvision.transforms as T
import torch
from collections import defaultdict
import json
import os

import sys
sys.path.insert(1, 'src/utility')
import scene_graph_constructor
from scene_graph_constructor import SceneGraphConstructor

class VisualGenomeDataset(Dataset):
    def __init__(
        self,
        scene_graph_json,
        image_data_json,
        colour_name_to_ID,
        colour_ID_to_name,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.constructor = SceneGraphConstructor(self)
        self.image_data_json = image_data_json
        self.scene_graph_json = scene_graph_json
        self.index = 0  # index for constructing the enum for categories

        # Create directory for scene graph .pt files if it doesn't exist
        os.makedirs("data/scene_graphs", exist_ok=True)

        # Load data
        # image_data = self.load_json(self.image_data_json)
        # scene_graph_data = self.load_json(self.scene_graph_json)
        image_data = self.load_partial_json(self.image_data_json)
        scene_graph_data = self.load_partial_json(self.scene_graph_json)

        # Initialize data structures
        self.initialize_lookup_tables(colour_name_to_ID, colour_ID_to_name)
        self.load_data_points(image_data)
        self.process_data_points(scene_graph_data)

    def initialize_lookup_tables(self, colour_name_to_ID, colour_ID_to_name):
        self.colour_enum = {
            # Two way lookup table mapping each colour attribute to a unique integer ID
            "colour_name_to_ID": defaultdict(int),
            "colour_ID_to_name": defaultdict(str),
        }

        # Update the default dictionaries with the provided dictionaries
        self.colour_enum["colour_name_to_ID"].update(colour_name_to_ID)
        self.colour_enum["colour_ID_to_name"].update(colour_ID_to_name)

        self.category_enum = (
            {  # Two way lookup table mapping each category to a unique integer ID
                "category_name_to_ID": defaultdict(int),
                "category_ID_to_name": defaultdict(str),
            }
        )

    def load_data_points(self, image_data, size=108077):
        """
        Processes each image in image_json, extracting its metadata and forming a
        new DataPoint instance. This DataPoint instance in stored in
        self.data indexed at the image's ID.
        """
        data_points = defaultdict(
            lambda: None)  # returns none if image_id is not a key
        # TODO: REMOVE THE INDEXING TO PROCESS EVERYTHING
        for item in image_data[:1]:
            image_id = item["image_id"]
            size = (item["width"], item["height"])
            url = item["url"]
            data_points[image_id] = self.DataPoint(image_id, url, size)
        self.data = data_points

    def process_data_points(self, scene_graph_data):
        """
        Constructs the scene graph for each DataPoint object and updates
        dataset's look up tables
        """
        for scene_graph in scene_graph_data[
            :1
        ]:  # TODO: REMOVE THE INDEXING TO PROCESS EVERYTHING
            image_id = scene_graph["image_id"]
            data_point = self.data[image_id]
            data_point.set_scene_graph(scene_graph, self.constructor)

    def add_category(self, category_name):
        if category_name not in self.category_enum["category_name_to_ID"]:
            self.category_enum["category_name_to_ID"][category_name] = self.index
            self.category_enum["category_ID_to_name"][self.index] = category_name
            self.index += 1
        return self.category_enum["category_name_to_ID"][category_name]

    class DataPoint:
        def __init__(self, image_id, url, size):
            self.id = image_id
            self.url = url
            self.size = size

            # One way look up table mapping each unique node in graph (object, subject, predicate) to its category ID
            self.category_mapping = defaultdict(int)

            # One way look up table mapping each unique node in graph (object, subject, predicate) to its colour ID
            self.colour_mapping = defaultdict(int)

        def set_scene_graph(self, scene_graph_data, constructor):
            (
                pytorch_graph,
                category_mapping,
                colour_mapping,
            ) = constructor.make_graph(scene_graph_data)

            self.graph = pytorch_graph
            self.set_mapping(category_mapping, colour_mapping)
            torch.save(
                pytorch_graph, f"data/scene_graphs/{self.id}.pt"
            )  # scene_graphs are saved by their image_id

        def set_mapping(self, node_ID_to_category_ID, node_ID_to_colour_ID):
            self.category_mapping = node_ID_to_category_ID
            self.colour_mapping = node_ID_to_colour_ID

        def get_url(self):
            return self.url

    def get_categories(self):
        return self.category_enum

    def get_colours(self):
        return self.colour_enum

    def len(self):
        return len(self.data)

    def get_data_point(self, image_id):
        # TODO: implement mechanism for checking if that item exists
        # defaultdict already returns a none
        return self.data[image_id]

    def get(self, image_id):
        data_point = self.get_data_point(image_id)
        # assuming `url` is a public attribute
        graph_path = f"data/scene_graphs/{data_point.url}.pt"
        data = torch.load(graph_path)
        return data

    @staticmethod
    def load_json(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error processing {file_path}: {e}")
            return None

    @staticmethod  # FOR TESTING
    def load_partial_json(file_path, max_items=5):
        items = []
        try:
            with open(file_path, "r") as f:
                # Assuming the top-level is an array in the JSON
                objects = ijson.items(f, "item")
                for index, item in enumerate(objects):
                    if index == max_items:
                        break
                    items.append(item)
            return items
        except (FileNotFoundError, ijson.JSONError) as e:
            print(f"Error processing {file_path}: {e}")
            return None


# Defining the most frequent colours within this dataset
colour_name_to_id = {
    "indistinct": 0,
    "white": 1,
    "black": 2,
    "blue": 3,
    "green": 4,
    "red": 5,
    "brown": 6,
    "yellow": 7,
    "gray": 8,
    "silver": 9,
    "orange": 10,
    "pink": 11,
    "tan": 12,
    "purple": 13,
    "gold": 14,
}
colour_id_to_name = {
    0: "indistinct",
    1: "white",
    2: "black",
    3: "blue",
    4: "green",
    5: "red",
    6: "brown",
    7: "yellow",
    8: "gray",
    9: "silver",
    10: "orange",
    11: "pink",
    12: "tan",
    13: "purple",
    14: "gold",
}


if __name__ == "__main__":
    scene_graph_json = "data/VisualGenome/scene_graphs.json"
    image_data_json = "data/VisualGenome/image_data.json"

    dset = VisualGenomeDataset(
        scene_graph_json=scene_graph_json,
        image_data_json=image_data_json,
        colour_name_to_ID=colour_name_to_id,
        colour_ID_to_name=colour_id_to_name,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    )
    # dset.process()
