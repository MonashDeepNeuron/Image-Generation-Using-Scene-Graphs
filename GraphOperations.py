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
        Return all edges coming out of a given node from a PyTorch Geometric Data object.
        
        Args:
        - data: A PyTorch Geometric Data object.
        - node: The node for which to retrieve the outgoing edges.
        
        Returns:
        - A list of tuples representing the outgoing edges from the node.
        """
        # Find indices where the source node matches the given node
        indices = (data.edge_index[0] == node).nonzero(as_tuple=True)[0]
        
        # Retrieve the destination nodes for these indices
        dest_nodes = data.edge_index[1][indices]
        
        # Form the edges and return
        return [(node, dest_node.item()) for dest_node in dest_nodes]