import torch
import torch.nn as nn
import numpy as np
import grep_youtill as g

class CustomGNN(nn.Module):
    """
    A single layer of scene graph convolution.
    """

    def __init__(self,
                 input_dim,
                 output_dim=None,
                 hidden_dim=[512, 64],
                 pooling="avg",
                 mlp_normalization="none",
                 num_layers = 3
                 ):
        # input_dim should be 3*dim of nodes

        if output_dim is None:
            output_dim = input_dim

        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim[0])
        self.fc2 = nn.Linear(
            in_features=hidden_dim[0], out_features=hidden_dim[1])
        self.out = nn.Linear(
            in_features=hidden_dim[1], out_features=output_dim)
        self.relu = nn.ReLU(inplace=True)

        self.num_layers = num_layers

    def forward(self, x):
        """
        forward implements one message passing layer 
        output: updated data object (?) or just matrix of nodes (?)
        params:
        - <Data.x> 
        dynamic_matrix = shallow copy of Data with predicate nodes replaced with flag variable

        for len(graph) iterations:
            for i, node enumerate() object node in graph: for each non-flagged entity in dynamic_matrix
                central_feature_vector_vector = [[]]
                outgoing edges = graph util function() 
                for each outgoing edge from outgoing edges
                    vector = [obj_node_feats, pred_node_feats, sub_nodes_feats]
                    out = FCN(vector)
                    central_feature_vector_vector.append(out) 
                new_feature_vecotr = pool(central_feature_vector_vector.entries())
                dynamic_matrix[i] = new_feature_vector
        """

        dynamic_matrix = x.copy() # shallow copy the data object, this will store the updated graph 

        # Extracting the indices of non-predicate nodes
        object_indices = [i if i, node[x] in enumerate(dynamic_matrix) != range(180,188)] # something like this, fix syntax later
        
        for layer in self.num_layers: # for each layer in our GNN
            for i in object_indices: # for each object node in our graph for the current layer
                node = dynamic_matrix.x[i] # extract the node features for that object -> i is the current object_indices
                central_node_feature_vector_vector = [] # this will store all the output vectors for each triplet aggregation
                
                # outgoing_connection has the form (predicate_node_ID, subject_node_ID)
                outgoing_connections = g.GraphOperations.find_subject_for_object(data = x, centre_node_ID = i)

                for connection in outgoing_connections:

                    # node.feature_vector = object feature vector
                    # edge[1].feature_vector = predicate feature vector
                    # edge[2].feature_vector = subject feature vector
                    vector = [node.feature_vector, connection[0].feature_vector, connection[1].feature_vector]

                    # pass it through our MLP to get our updated vector for the current triplet aggregation
                    updated_vector = self.fcn_pass(vector)
                    
                    # append the 
                    central_node_feature_vector_vector.append(updated_vector)
                updated_central_node_feature_vector = np.average(central_node_feature_vector_vector, axis=0)
                dynamic_matrix[i] = updated_central_node_feature_vector

        return dynamic_matrix # input to layout generator

    def fcn_pass(self, vector):
        x = self.relu(self.fc1(vector))
        x = self.relu(self.fc2(x))
        out = self.out(x)

        return out
