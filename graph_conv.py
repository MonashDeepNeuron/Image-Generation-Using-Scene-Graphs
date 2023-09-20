import torch
import torch.nn as nn


class CustomGraphConv(nn.Module):
    """
    A single layer of scene graph convolution.
    """

    def __init__(self,
                 input_dim,
                 output_dim=None,
                 hidden_dim=[512, ],
                 pooling="avg",
                 mlp_normalization="none",
                 ):
        pass

    def forward():
        """
        forward implements one message passing layer 
        output: updated data object (?) or just matrix of nodes (?)

        matrix = [[]*number of non-predicate nodes]
        for len(graph) iterations:
            for each object node in graph:
                central_feature_vector_vector = [[]]
                for each outgoing edge from object node (the predicate node and the subject node)
                    vector = [obj_node_feats, pred_node_feats, sub_nodes_feats]
                    out = FCN(vector)
                    central_feature_vector_vector.append(out) 
                new_feature_vecotr = pool(central_feature_vector_vector.entries())
                matrix.append

                

        """


class CustomGNN(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):

        # x -> data object
        out1 = CustomGraphConv(x)
