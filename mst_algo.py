import networkx as nx
from networkx.algorithms import tree
import math
import torch
import geometric 

from enum import Enum
def construct_node_matrix(self, nodes):
    '''
    nodes: node_list[x] returns the class ID of node x

    This method constructs the adjacency matrix and node 
    feature vectors for a given obj (list of nodes in the 
    form [vertex ID, position vector]). 

    1. The centres of all nodes are calculated and stored
    in 'nodes_centers'. This simplifies each node to a single 
    coordinate . 

    2. 'Edges' are made between each pair of nodes with weight
    being the Euclidean distance between their centres. These are
    added to a NetworkX graph, along with the list of vertex IDs

    3. Kruskals is run on this graph, returning the list of edges
    'edge_list' with each entry being an edge in the form
    (u,v,dist). These edges make up the minimum spanning tree
    (edge_list is the adjacency list for Data graph, remove the dist info)

    4. For each u, v in each edge in 'edge_list', construct a new 
    relationship node based on u and v's relative positions, add
    this node to node_feat tensor in graph and add edge to edge_list 
    in graph 

    5. Add all nodes to the Data graph (with coordinates and object class ID
    in their node feature vector)

    6. Graph canonlicalisation  
    '''
    
    '''
    Constructing node_centres. node_centres[i] returns a tuple (x,y) relating to the centre of object i 
    '''
    node_centres = []
    _, MH, MW = masks.size()
    for i, node in enumerate(nodes):
        x0, y0, x1, y1 = boxes[i]
        mask = (masks[i] == 1) 
        xs = torch.linspace(x0, x1, MW).view(1, MW).expand(MH, MW)
        ys = torch.linspace(y0, y1, MH).view(MH, 1).expand(MH, MW)
        if mask.sum() == 0: # Compute means of objects, if object mask sums to zero take midpoint
            mean_x = 0.5 * (x0 + x1)
            mean_y = 0.5 * (y0 + y1)
        else:
            mean_x = xs[mask].mean()
            mean_y = ys[mask].mean()
        node_centres.append([mean_x, mean_y])
    node_centres = torch.FloatTensor(node_centres)

    '''
    Applying Kruskals Algorithm to generate the minimum spanning scene graph based on euclidean distance 
    '''
    # Create a new graph
    G = nx.Graph()

    # Add nodes to graph. Each node is it's integer vertex ID
    for index, currentObject in enumerate(nodes):
        G.add_node(index)

    num_nodes = len(nodes)
    # Create list of edges in form [u,v,euclidean distance]
    for i in range(num_nodes): #for each node
        for j in range(i+1, num_nodes): # to avoid duplicate edges
            # dist calculates the euclidean distance between two object centres
            dist = (lambda u, v :math.sqrt((u[0]-v[0])^2+(u[1]-v[1])^2)) 
            edge_representation = (i,j,dist(obj_centres(i), obj_centres(j))) 
            G.add_edge(edge_representation)

    # Find the minimum spanning tree using Kruskal's algorithm
    mst = tree.minimum_spanning_edges(G, algorithm="kruskal", data=False)
    edgelist = list(mst) #### this is our triples
    
    # Constructing adjacency list (edge list)
    simplify_edge = lambda edge : [edge[0], edge[1]]
    adjacency_list = edgelist.map(simplify_edge, edgelist) 

    # Constructing node feature vectors: each entry is [class ID, node centre]
    feature_vectors = torch.tensor()
    for i in range (num_nodes):
        new_feature_vector = torch.tensor([nodes[i], node_centres[i]], dim=0)
        feature_vectors.cat(new_feature_vector,0)


    '''
    Adding SuperNode to adjacency list and 

    Supernode coordinates are always (0,0,1,1), where 0,0 represents the bottom left of the image and 1,1 represents
    the top right of the image. This is always the case!
    '''
    # Add __in_image__ triples
    supernode_ID = nodes.size(0) # Assigning a unique node ID to the supernode
    in_image_classID = self.vocab['pred_name_to_idx']['__in_image__']

    # Adding supernode feature vector to feature_vectors
    supernode_feature_vector = torch.tensor([in_image_classID, position],[0.5,0.5])
    feature_vectors.cat(supernode_feature_vector,[0.5,0.5])
    for node_index in range(nodes.size(0)-1):
        
    # TODO: MAKE NODES FOR THE IN_IMAGE RELATIONSHIPS
    # Adding edge between supernode and every node to adjacency list
    for i in range(supernode_ID - 1):
        adjacency_list.append([supernode_ID, i])
    feature_vectors.cat(new_feature_vector,0)
    '''
    Adding Predicate Nodes: For each edge, make it a node with a nodeID. 
    '''
    nodeID = supernode_ID+1  # Starting node ID to give to predicate nodes
    midpoint = lambda p1, p2: ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    for predicate in edgelist: # Predicate has the form [subjectID, objectID, euclideanDistance]
        subjectID  = predicate[0]
        objectID  = predicate[1]
        
        # Find the midpoint vector m from node_centres[u] node_centres[v]
        position = midpoint(node_centres[subjectID], node_centres[objectID])
        # Determine the class ID by determining the relationship between the subject and object
        classID = determine_relationship(subjectID, objectID)

        # Add predicate node to node feature vectors tensor 
        new_feature_vector = torch.tensor([classID, position],0)
        feature_vectors.cat(new_feature_vector,0)

        # Add edges from subject --> predicate and predicate --> object
        sub2pred = [subjectID, nodeID]
        pred2obj = [nodeID, objectID]
        adjacency_list.append(sub2pred)
        adjacency_list.append(pred2obj)
        nodeID+=1

    # x is node feat
    coco_scene_grah = Data(x=feature_vectors, edge_index=adjacency_list)

    return coco_scene_grah


    '''
    Writing to Pytorch geomtric graph representation
    Graph Representation: A graph is represented using two primary tensors
    x: A tensor that contains the node features. Its size is [num_nodes, num_node_features].
    edge_index: A tensor of shape [2, num_edges], describing the connectivity of the graph.
    Each column represents an edge. If edge_index[:, i] = [src, dest],
    then node src is connected to node dest.
    '''
    
    @staticmethod
    def determine_relationship(s,o):
        """
        determine_relationship determines the spatial relationship between subject and object, and returns
        a relationship listed in Direction
        """
        sx0, sy0, sx1, sy1 = boxes[s] ## subject corners
        ox0, oy0, ox1, oy1 = boxes[o] ## object corners 
        d = node_centres[s] - node_centres[o]
        theta = math.atan2(d[1], d[0])  
  
        LEFT_BOUND = -3 * math.pi / 4
        RIGHT_BOUND = 3 * math.pi / 4
        theta = math.atan2(d[1], d[0])  # Calculate angle

        # Check for surrounding and inside conditions
        if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
            return 'SURROUNDING'
        if sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
            return 'INSIDE'

        # Determine direction based on angle
        if theta >= RIGHT_BOUND or theta <= LEFT_BOUND:
            return 'LEFT '
        if LEFT_BOUND <= theta < -math.pi / 4:
            return 'ABOVE'
        if -math.pi / 4 <= theta < math.pi / 4:
            return 'RIGHT'
        if math.pi / 4 <= theta < RIGHT_BOUND:
            return 'BELOW'

    class Direction(Enum):
        LEFT = 183
        RIGHT = 184
        ABOVE = 185
        BELOW = 186
        INSIDE = 187
        SURROUNDING = 188
