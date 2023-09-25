import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import tree
import math
import torch
from enum import Enum
from torch_geometric.data import Data
from typing import List, Dict


class SceneGraphConstructor:
    def __init__(self, dataset):
        self.dataset = dataset

    def make_graph(self, scene_graph):
        """
        Construct a graph from a given scene graph.

        Parameters:
        - scene_graph: A dictionary representing the scene graph.

        Returns:
        - vg_scene_graph: A PyTorch Geometric Data object formed from the feature_vector and adjacency_list.
        - category_mapping: A dictionary mapping each node to its category ID.
        - colour_mapping: A dictionary mapping each node to its colour ID.

        - <array> node_feature_vector: [node_id, x, y, h, w, category_id, colour_id, id] A single entry in all_feature_vectors
        representing the feature vector of one node.
        """

        all_feature_vectors = []
        category_mapping = {}
        colour_mapping = {}
        node_id = 0  # Tracking the assignment of node IDs
        edge_list = []

        # Process objects in the scene graph
        for object_dict in scene_graph["objects"]:
            (
                new_feature_vector,
                category_id,
                colour_id,
            ) = self.make_object_node_feature_vector(object_dict, node_id)
            all_feature_vectors.append(new_feature_vector)
            category_mapping[node_id] = category_id
            colour_mapping[node_id] = colour_id
            node_id += 1

        # Helper function to find the index of a vector with a given ID
        def _find_vector_index_by_id(vectors, id_value):
            return next(
                (i for i, vec in enumerate(vectors)
                 if vec[7] == id_value), None
            )

        # Process predicates in the scene graph
        for predicate_dict in scene_graph["relationships"]:
            object_index = _find_vector_index_by_id(
                all_feature_vectors, predicate_dict["object_id"]
            )
            subject_index = _find_vector_index_by_id(
                all_feature_vectors, predicate_dict["subject_id"]
            )

            (
                predicate_feature_vector,
                category_id,
                colour_id,
            ) = self.make_predicate_node_feature_vector(
                object_vector=all_feature_vectors[object_index],
                subject_vector=all_feature_vectors[subject_index],
                predicate_dict=predicate_dict,
                node_id=node_id,
            )

            # Update edge list and feature vectors
            edge_list.extend(
                [[object_index, node_id], [node_id, subject_index]])
            all_feature_vectors.append(predicate_feature_vector)
            category_mapping[node_id] = category_id
            colour_mapping[node_id] = colour_id
            node_id += 1

        all_feature_vectors = [inner[:-1] for inner in all_feature_vectors]
        # # Convert feature vectors and edge list to tensors
        feature_vectors_tensor = torch.stack(all_feature_vectors, dim=0)
        edge_list_tensor = torch.tensor(edge_list).t().contiguous()

        # Construct and return the vg_scene_graph Data object
        vg_scene_graph = Data(x=feature_vectors_tensor,
                              edge_index=edge_list_tensor)
        print(f"category mapping: {category_mapping}")
        print(f"colour mapping: {colour_mapping}")
        return vg_scene_graph, category_mapping, colour_mapping

    def _get_category_id(self, item_dict, key_name, default_name):
        """
        Helper function to get the category_id based on a given dictionary, key_name, and a default_name.
        """
        if "synsets" in item_dict and item_dict["synsets"]:
            category_name = item_dict["synsets"][0]
        else:
            category_name = item_dict.get(key_name, default_name)

        category_id = self.dataset.add_category(
            category_name
        )  # Add to enum if doesn't exist
        return category_id

    def make_object_node_feature_vector(self, object_dict, node_id):
        x, y, h, w = (
            object_dict["x"],
            object_dict["y"],
            object_dict["h"],
            object_dict["w"],
        )
        object_id = object_dict["object_id"]
        category_id = self._get_category_id(object_dict, "name", None)

        colour_name = object_dict.get("attributes", [None])[0]

        if colour_name:
            colour_name = (
                colour_name.strip()
            )  # remove trailing and leading spaces surrounding colour
        # returns dictionary of dictionaries, must index subdictionary
        colour_enum = self.dataset.get_colours()
        colour_id = (
            list(colour_enum["colour_name_to_ID"]).index(
                colour_name) if colour_name in colour_enum["colour_name_to_ID"] else 0
        )

        new_feature_vector = torch.tensor(
            [node_id, x, y, h, w, category_id,
                colour_id, object_id], dtype=torch.float
        )
        return new_feature_vector, category_id, colour_id

    def make_predicate_node_feature_vector(
        self, subject_vector, object_vector, predicate_dict, node_id
    ):
        x, y = (subject_vector[0] + object_vector[0]) / 2, (
            subject_vector[1] + object_vector[1]
        ) / 2  # Calculate position as the midpoint of object and subject

        # Convert x and y tensors to integers
        x_int = int(x.item())
        y_int = int(y.item())

        category_id = self._get_category_id(predicate_dict, "predicate", None)
        predicate_id = predicate_dict["relationship_id"]

        new_feature_vector = torch.tensor(
            [node_id, x, y, 0, 0, category_id, 0, predicate_id], dtype=torch.float
        )

        return (
            new_feature_vector,
            category_id,
            0,
        )  # 0 corresponds to indistinct colour for predicate
