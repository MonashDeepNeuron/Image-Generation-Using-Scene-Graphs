import torch
import torch.nn as nn
import numpy as np
import grep_youtill as g


class CustomGNN(nn.Module):
    """
    A single layer of scene graph convolution.
    """

    def __init__(
        self,
        input_dim,
        output_dim=None,
        hidden_dim=[512, 64],
        pooling="avg",
        mlp_normalization="none",
        num_layers=1,
    ):
        super(CustomGNN, self).__init__()
        # input_dim should be 3*dim of nodes

        if output_dim is None:
            output_dim = input_dim

        self.fc1 = nn.Linear(in_features=input_dim*3,
                             out_features=hidden_dim[0])
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

        dynamic_matrix = (
            x.clone()
        )  # shallow copy the data object, this will store the updated graph

        # Extracting the indices of non-predicate nodes
        object_indices = []
        node_id_to_node_idx = {}

        # "i" would describe the current node
        for i, node in enumerate(dynamic_matrix.x):
            # if not relationship or super node, is an object node
            if node[1] < 183 and node[1] != 0:
                object_indices.append(i)

            # we really don't need this idk why i used this lmao
            node_id_to_node_idx.update({int(node[0].item()): i})

        output_object_feature_vectors = []  # final object feature vectors

        for i in range(self.num_layers):  # for each layer in our GNN
            # data structure to store updated feature vectors for each object
            vector_of_updated_node_feature_vectors = []
            for i in object_indices:  # for each object node in our graph for the current layer
                node = dynamic_matrix.x[
                    i, :
                ]  # extract the node features for that object -> i is the current object_indices
                central_node_feature_vector_vector = (
                    []
                )  # this will store all the output vectors for each triplet aggregation

                # outgoing_connection has the form (predicate_node_ID, subject_node_ID)
                outgoing_connections = g.GraphOperations.find_subject_for_object(
                    data=x, centre_node_ID=i
                )

                for connection in outgoing_connections:
                    if len(connection) == 0:  # if endpoint node, skip
                        continue
                    # print(node_id_to_node_idx)
                    predicate = dynamic_matrix.x[node_id_to_node_idx[connection[0].item(
                    )], :]
                    subject = dynamic_matrix.x[node_id_to_node_idx[connection[1].item(
                    )], :]

                    vector = torch.cat((node, predicate, subject), dim=-1)

                    # pass it through our MLP to get our updated vector for the current triplet aggregation
                    updated_vector = self.fcn_pass(vector).detach().numpy()
                    # append the updated central vector for that triplet aggregation
                    central_node_feature_vector_vector.append(updated_vector)

                # again, if endpoint node we don't udpate its representation, else update representation
                # and store this in data structure (only update node feature vectors after all nodes have
                # been convolved)
                if len(central_node_feature_vector_vector) != 0:
                    updated_central_node_feature_vector = np.average(
                        central_node_feature_vector_vector, axis=0
                    )
                    vector_of_updated_node_feature_vectors.append(
                        updated_central_node_feature_vector)
                else:
                    # end point nodes are not updated in directed graph
                    vector_of_updated_node_feature_vectors.append(
                        np.asarray(node))

            for i in object_indices:
                dynamic_matrix.x[i] = torch.tensor(
                    vector_of_updated_node_feature_vectors[i])  # update node feature vectors after all nodes have been convolved

        for i in object_indices:
            # only return object node feature vectors
            output_object_feature_vectors.append(dynamic_matrix.x[i])
        return output_object_feature_vectors  # input to layout generator

    def fcn_pass(self, vector):
        x = self.relu(self.fc1(vector))
        x = self.relu(self.fc2(x))
        out = self.out(x)
        return out


if __name__ == "__main__":
    gnn = CustomGNN(input_dim=6, output_dim=6)
    data = torch.load("please_god.pt")
    output = gnn(data)
    # print(output)
