# Bringing in modules/dependencies/etc:
import os
import random
import math
import json

from collections import defaultdict
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import PIL

import torch_geometric
from torch_geometric.data import Dataset, download_url

# import networkx as nx
# from networkx.algorithms import tree

import scene_graph_constructor
from scene_graph_constructor import SceneGraphConstructor


class VisualGenomeDataset(Dataset):
    """
    VisualGenomeDataset represents our Visual Genome dataset as an object. It inherits from PyTorch Geometric's "Dataset"
    class, which requires us to implement the len() and get() methods.

    The main methods of interest:
        1. __init__(): In this __init__ function, we construct all our data structures to store important image data, including
        image id and the corresponding image url, as well as symmetric lookup tables for name-idx pairs and general arrays for
        storing scene graphs.
        2. process(): This method utilises the SceneGraphConstructor instance to construct PyG Data objects that represent the
        encoded scene graphs in the scene_graph_json file.
    """

    def __init__(
        self,
        scene_graph_json,
        image_data_json,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.image_data_json = image_data_json  # path to image metadata json file
        self.scene_graph_json = scene_graph_json  # path to scene graph json file

        self.image_data = None
        self.scene_graph_data = None

        self.colour_whitelist = [
            "red",
            "blue",
            "green",
            "yellow",
            "white",
            "silver",
            "pink",
            "purple",
        ]  # some objects have colour attributes

        # Read image metadata (image id, url)
        try:
            with open(self.image_data_json, "r") as image_data_file:
                image_data = json.load(image_data_file)
        except FileNotFoundError:
            print(f"The file '{self.image_data_json}' does not exist.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

        # Read scene graphs (relationships, object/subject)
        try:
            with open(self.scene_graph_json, "r") as scene_graph_file:
                scene_graph_data = json.load(scene_graph_file)
        except FileNotFoundError:
            print(f"The file '{self.scene_graph_json}' does not exist.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

        # Initialise basic data structures
        self.all_image_ids = []  # this stores all image ids

        self.image_id_to_url = (
            {}
        )  # map image id to url (so that we can retrieve the actual image)
        self.image_id_to_size = (
            {}
        )  # map image id to size (FUTURE: could need to revert image to original size)

        self.scene_graph_array = (
            []
        )  # array to store all scene graphs from scene_graph json

        # Initalise symmetric lookup tables
        # for class name to class index (REDUNDANT, but good for testing lol)
        self.all_object_categories = {
            "class_name_to_class_idx": {},
            "class_idx_to_class_name": {},
        }
        # for relationship name to relationship class index (REDUNDANT)
        self.all_relationship_categories = {
            "rel_name_to_rel_idx": {},
            "rel_idx_to_rel_name": {},
        }
        # for object/relationship class to object/relationship node index (IN USE)
        self.all_node_categories = {
            "node_name_to_node_idx": {},
            "node_idx_to_node_name": {},
        }
        # for colour name to colour index (IN USE)
        self.all_colours = {
            "colour_name_to_colour_idx": {},
            "colour_idx_to_colour_name": {},
        }

        # Get image metadata for each image
        if image_data is not None:
            for item in image_data:
                image_id = item["id"]
                file_url = item["url"]
                width = item["width"]
                height = item["height"]
                self.all_image_ids.append(image_id)
                self.image_id_to_url[image_id] = file_url
                self.image_id_to_size[image_id] = (width, height)

        # Get scene graph data for each iamge
        if scene_graph_data is not None:
            node_idx = 1
            colour_idx = 1
            for item in scene_graph_data:
                self.scene_graph_array.append(item)
                relationships = item["relationships"]
                for rel in relationships:
                    try:
                        rel_list = rel["synsets"][0]
                    except:
                        continue
                    for rel_name in rel_list:
                        if not self.all_relationship_categories["rel_name_to_rel_idx"]:
                            # Key doesn't exist, so add it with the associated value
                            self.all_relationship_categories[
                                "rel_name_to_rel_idx"
                            ].update(
                                {rel_name: node_idx}
                            )  # relationship_idx
                            self.all_relationship_categories[
                                "rel_idx_to_rel_name"
                            ].update({node_idx: rel_name})
                            self.all_node_categories["node_name_to_node_idx"].update(
                                {rel_name: node_idx}
                            )
                            self.all_node_categories["node_idx_to_node_name"].update(
                                {node_idx: rel_name}
                            )
                            node_idx += 1
                            continue
                        if (
                            rel_name
                            not in self.all_relationship_categories[
                                "rel_idx_to_rel_name"
                            ]
                        ):
                            # Key doesn't exist, so add it with the associated value
                            self.all_relationship_categories[
                                "rel_name_to_rel_idx"
                            ].update(
                                {rel_name: node_idx}
                            )  # relationship_idx
                            self.all_relationship_categories[
                                "rel_idx_to_rel_name"
                            ].update({node_idx: rel_name})
                            self.all_node_categories["node_name_to_node_idx"].update(
                                {rel_name: node_idx}
                            )
                            self.all_node_categories["node_idx_to_node_name"].update(
                                {node_idx: rel_name}
                            )
                            node_idx += 1
                objects = item["objects"]
                for obj in objects:
                    try:
                        obj_list = obj["synsets"]
                    except:
                        continue
                    for obj_name in obj_list:
                        if not self.all_object_categories["class_name_to_class_idx"]:
                            # Key doesn't exist, so add it with the associated value
                            self.all_object_categories[
                                "class_name_to_class_idx"
                            ].update({obj_name: node_idx})
                            self.all_object_categories[
                                "class_idx_to_class_name"
                            ].update({node_idx: obj_name})
                            self.all_node_categories["node_name_to_node_idx"].update(
                                {obj_name: node_idx}
                            )
                            self.all_node_categories["node_idx_to_node_name"].update(
                                {node_idx: obj_name}
                            )
                            node_idx += 1
                            continue
                        if (
                            obj_name
                            not in self.all_object_categories["class_name_to_class_idx"]
                        ):
                            # Key doesn't exist, so add it with the associated value
                            self.all_object_categories[
                                "class_name_to_class_idx"
                            ].update({obj_name: node_idx})
                            self.all_object_categories[
                                "class_idx_to_class_name"
                            ].update({node_idx: obj_name})
                            self.all_node_categories["node_name_to_node_idx"].update(
                                {obj_name: node_idx}
                            )
                            self.all_node_categories["node_idx_to_node_name"].update(
                                {node_idx: obj_name}
                            )
                            node_idx += 1
                    try:
                        obj_attributes = obj["attributes"]
                    except:
                        continue
                    if obj_attributes is not None:
                        for attribute in obj_attributes:
                            if (
                                attribute in self.colour_whitelist
                                and not self.all_colours["colour_name_to_colour_idx"]
                            ):
                                self.all_colours["colour_name_to_colour_idx"].update(
                                    {attribute: colour_idx}
                                )
                                self.all_colours["colour_idx_to_colour_name"].update(
                                    {colour_idx: attribute}
                                )
                                colour_idx += 1
                                continue
                            if (
                                attribute in self.colour_whitelist
                                and attribute
                                not in self.all_colours["colour_name_to_colour_idx"]
                            ):
                                self.all_colours["colour_name_to_colour_idx"].update(
                                    {attribute: colour_idx}
                                )
                                self.all_colours["colour_idx_to_colour_name"].update(
                                    {colour_idx: attribute}
                                )
                                colour_idx += 1

        self.constructor = SceneGraphConstructor()

    @property
    def raw_file_names(self):
        return None  # path to all images

    @property
    def processed_file_names(self):
        return None  # path to prcoessed graphs

    def process(self) -> None:
        Path("scene_graphs").mkdir(parents=True, exist_ok=True)
        # print(self.scene_graph_array)
        for graph in self.scene_graph_array[0:2]:
            graph_data_object = self.constructor.construct_scene_graphs(
                scene_graph=graph,
                category_dict=self.all_node_categories["node_name_to_node_idx"],
                colour_dict=self.all_colours["colour_name_to_colour_idx"],
            )
            # SAVE EACH SCENE GRAPH USING ITS IMAGE ID

            # Save the file in the directory
            # torch.save(graph_data_object, "scene_graphs/please_god.pt")

            torch.save(graph_data_object, "please_god.pt")
            break

    def len(self):
        return len(self.scene_graph_array)

    def get(self, idx):
        # Retrieve the file name using the index
        data = torch.load("please_god.pt")
        return data


if __name__ == "__main__":
    # image_dir = "/Users/nyankyaw/Documents/UNI/YEAR4/MDN/Image-Generation-Using-Scene-Graphs/data/VisualGenome/images"
    scene_graph_json = "scene_graphs.json"
    image_data_json = "image_data.json"

    dset = VisualGenomeDataset(
        scene_graph_json=scene_graph_json,
        image_data_json=image_data_json,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    )
    dset.process()
