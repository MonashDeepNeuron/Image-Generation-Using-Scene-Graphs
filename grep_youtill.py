class GraphOperations:
    @staticmethod
    def edges_to_adjacency_matrix(edges):
        """
        Returns the adjacency matrix for a given edge list 
        """
        # Determine the number of vertices
        n = max(max(edge) for edge in edges) + 1

        # Initialize the adjacency matrix with zeros
        matrix = [[0] * n for _ in range(n)]

        # Populate the adjacency matrix
        for i, j in edges:
            matrix[i][j] = 1

        return matrix

    @staticmethod
    def edges_from_node(data, node):
        """
        Return all subject nodes connected to a given node by an outgoing edge 
        from a PyTorch Geometric Data object.
        Returns a list of all subject nodes connected to the object node via an outgoing edge 

        (object) --> (predicate) --> (subject)
        Args:
        - data: A PyTorch Geometric Data object.
        - nodeID: The nodeID for which to retrieve the outgoing edge.

        Returns:
        - A list of subject node IDs representing the adjacent subject nodes from nodeID
        """
        # Find indices where the source node matches the given node
        indices = (data.edge_index[0] == node).nonzero(as_tuple=True)[0]
        # indices = (data.edge_list[0] == node).nonzero(as_tuple=True)[0]

        # Retrieve the destination nodes for these indices
        dest_nodes = data.edge_index[1][indices]

        # Find the indices of the

        # Form the edges and return
        return [(node, dest_node.item()) for dest_node in dest_nodes]

    @staticmethod
    def find_subject_for_object(data, centre_node_ID):
        """
        Return all adjacent (predicate_node_ID, subject_node_ID)'s connected to a given node by an outgoing edge.
        (object_node) --> (predicate_node) --> (subject_node)

        Parameters:
        - data: A PyTorch Geometric Data object.
        - nodeID: The nodeID for which to retrieve the outgoing connections.

        Returns:
        - connections: list of (predicate node ID, subject node ID) representing the adjacent subject nodes from nodeID
        """
        connections = []
        # Find indices (which edges) where the source node matches the given node
        pred_indices = (data.edge_index[0] == centre_node_ID).nonzero(
            as_tuple=True)[0]
        # indices = (data.edge_list[0] == node).nonzero(as_tuple=True)[0]

        # Retrieve the predicate nodes for these edge indices
        pred_nodes = data.edge_index[1][pred_indices]

        # for each pred node, retrieve its corresponding subject node
        for predicate_node in pred_nodes:
            # Find the edge which connects the predicate node to the subject node
            subject_index = (data.edge_index[0] == predicate_node).nonzero(
                as_tuple=True)[0]

            # Retrieve the subject nodes for these edgeindices
            subject_node = data.edge_index[1][subject_index]

            # Add this subject node
            connections.append([predicate_node, subject_node])
        return connections

    # @staticmethod
    # def sort_feature_vector_array(data)
