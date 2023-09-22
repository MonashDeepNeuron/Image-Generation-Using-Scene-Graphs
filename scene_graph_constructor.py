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


# from GraphOperations import edges_to_adjacency_matrix
class SceneGraphConstructor:
    """
    SceneGraphConstructor instances implement the construct_scene_graphs method which which returns a PyTorch
    Geometric Data object encoding the scene graph information (object nodes & predicate nodes).
    """

    def construct_scene_graphs(self, scene_graph, category_dict, colour_dict):
        print("in scene graph constructor")
        # def construct_scene_graphs(self, scene_graph: dict, category_dict: dict, colour_dict: dict) ->  Data:
        """
        construct_scene_graph constructs and returns a PyTorch Geometric Data object.

        params:
        - <dict> scene_graph: one entry of the scene_graphs array (from scene_graphs json file)
        - <dict> category_dict: all_categories['node_name_to_idx']
        - <dict> colour_dict: all_colours['colour_name_to_idx']

        returns:
        - <PyTorch Geometric Data Object>: 'vg_scene_graph' formed from feature_vector and adjacency_list

        Important variables:
        - <tensor> all_feature_vectors: The node feature matrix. This is used to represent the features
        associated with each node in the graph. Each row of the matrix corresponds to a node,
        and each column corresponds to a feature. The entry Xij represents the value of the
        j-th feature for the i-th node.

        - <array> node_feature_vector: [x, y, h, w, category_id, colour_id, id] A single entry in all_feature_vectors
        representing the feature vector of one node.
        """

        """
        Process all objects in scene graphs and add each to the node_feature_vector
        """

        all_feature_vectors = []  # Create an array holding all node feature vectors

        for object_dict in scene_graph["objects"]:
            new_feature_vector = self.make_object_node_feature_vector(
                object_dict, category_dict, colour_dict
            )
            all_feature_vectors.append(new_feature_vector)

        """
        Retrieve the corresponding object and subject node feature vectors for the construction
        of each predicate node (for use in defining its x, y coordinate)

        pseudocode:
        1. For each predicate dict, retrieve the subject and object feature vectors by searching
        through all_feature_vectors[6] (node_feature_vector: [x, y, h, w, category_id, colour_id, id])
        2. Construct the predicate feature vector and add to all_feature_vectors
        3. Add the directed edges between object --> predicate and predicate --> subject to the edge_list 

        This all relies on the order of node feature vectors being preserved (as each node is distinguished
        by their index in all_node_feature_vectors. This is their node_ID)
        """
        edge_list = []

        # Keeping track of the vertex_ID to assign to the current predicate
        predicate_index = len(all_feature_vectors)
        for predicate in scene_graph["relationships"]:
            object_id = predicate["object_id"]
            subject_id = predicate["subject_id"]

            # Finding the corresponding object and subject nodes in all_feature_vectors
            object_vector, object_index = next(
                (
                    (vec, i)
                    for i, vec in enumerate(all_feature_vectors)
                    if vec[6] == object_id
                ),
                (None, None),
            )
            subject_vector, subject_index = next(
                (
                    (vec, i)
                    for i, vec in enumerate(all_feature_vectors)
                    if vec[6] == subject_id
                ),
                (None, None),
            )

            # Constructing the predicate node feature vector
            predicate_node = self.make_predicate_node_feature_vector(
                object_vector=object_vector,
                subject_vector=subject_vector,
                predicate_dict=predicate,
                category_dict=category_dict,
            )

            # Adding the object --> predicate and predicate --> subject edges
            object_edge = [object_index, predicate_index]
            subject_edge = [predicate_index, subject_index]
            edge_list.append(object_edge, subject_edge)

            current_predicate_index += 1

        """
        Construct and return the vg_scene_graph Data object
        """
        # Remove the last entry of each inner feature vector (the object or predicate ID)
        trimmed_feature_vectors = [inner[:-1] for inner in all_feature_vectors]

        # Convert edge_list and feature_vectors to tensors
        feature_vectors_tensor = torch.tensor(trimmed_feature_vectors)
        edge_list_tensor = torch.tensor(edge_list).t().contiguous()

        # Construct the data object and return
        vg_scene_graph = Data(x=feature_vectors_tensor, edge_index=edge_list_tensor)
        print(vg_scene_graph)
        return vg_scene_graph

    @staticmethod
    def make_object_node_feature_vector(
        object_dict, all_node_categories, all_colour_categories
    ):
        # def make_object_node_feature_vector(object_dict: dict, all_node_categories: dict,
        # all_colour_categories: dict) -> torch.Tensor:
        """
        object_dict example structure:
        {
            "synsets": [
                "pole.n.01"
            ],
            "h": 438,
            "object_id": 1023847,
            "names": [
                "pole"
            ],
            "w": 78,
            "attributes": [
                "brown"
            ],
            "y": 34,
            "x": 405
        }
        """
        x = object_dict["x"]
        y = object_dict["y"]
        h = object_dict["h"]
        w = object_dict["w"]
        object_id = object_dict["object_id"]

        # Assuming the first name in the 'synsets' list is the main category for the node
        category_name = object_dict["synsets"][0]
        category_id = list(all_node_categories).index(category_name)

        # Assuming the first colour is the colour of the object
        colour_name = object_dict["attributes"][0]
        if colour_name in all_colour_categories:
            colour_id = list(all_colour_categories).index(colour_name)

        # If the attribute is not whitelisted and exists as a key in all_colour_categories, assign 0
        else:
            colour_id = 0

        # Construct the new_feature_vector
        new_feature_vector = torch.tensor(
            [x, y, h, w, category_id, colour_id, object_id], dtype=torch.float
        )

        return new_feature_vector

    @staticmethod
    def make_predicate_node_feature_vector(
        subject_vector, object_vector, predicate_dict, all_node_categories
    ):
        # def make_predicate_node_feature_vector(subject_vector: List[float], object_vector: List[float],
        #                                        predicate_dict: dict, all_node_categories: dict) -> torch.Tensor:
        """
        predicate_dict example structure:
        {
            "synsets": [
                "wear.v.01"
            ],
            "predicate": "wears",
            "relationship_id": 15947,
            "object_id": 5071,
            "subject_id": 1023838
        }
        """

        # Finding the midpoint of the object and subject nodes to determine the x, y position of the predicate node
        x = (subject_vector[0] + object_vector[0]) / 2
        y = (subject_vector[1] + object_vector[1]) / 2

        # The bounding box of a predicate node has no area
        w = 0
        h = 0

        predicate_id = predicate_dict["relationship_id"]

        # Assuming the first synset in the 'synsets' list is the primary category for the predicate
        category_name = predicate_dict["synsets"][0]
        category_id = list(all_node_categories).index(category_name)

        # The colour_ID of a predicate node is always 0
        colour_id = 0

        new_feature_vector = torch.tensor(
            [x, y, w, h, category_id, colour_id, predicate_id], dtype=torch.float
        )

        return new_feature_vector
