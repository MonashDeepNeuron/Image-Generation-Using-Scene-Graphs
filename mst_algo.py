import networkx as nx
from networkx.algorithms import tree
import math
import torch
from enum import Enum
from torch_geometric.data import Data

'''
Current COCO-Stuff Data Loading Progress:
nodes: node_list[x] returns the class ID of node x

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

        6. Graph canonicalisation  

'''
'''
Vertex IDs:
    1...len(nodes)-1   len(nodes)       len(nodes)+1.....
|---- object nodes ----|supernode| ---- predicate nodes ----|
'''
class SceneGraphConstructor:
    '''
    SceneGraphConstructor instances implement the construct_node_matrix method which which returns a PyTorch
     Geometric Data object encoding the scene graph information (object nodes, supernode & predicate nodes).
    '''
    class Direction(Enum):
        
        LEFT = 183
        RIGHT = 184
        ABOVE = 185
        BELOW = 186
        INSIDE = 187
        SURROUNDING = 188

    def __init__(self, vocab, boxes):
        self.vocab = vocab
        self.boxes = boxes

    def construct_scene_graph(self, nodes, boxes, masks):
        '''
        construct_scene_graph constructs the PyTorch Geometric Data object from a given list of nodes
         (each ) and node 

        params:
        - <Tensor of ClassIDs indexed by Vertex IDs> nodes: contains all objects in a given image
        - <Tensor of bounding boxes indexed by Vertex IDs> boxes
        - <Tensor of segmentation masks indexed by Vertex IDs> masks
        feature vectors for a given image (list of nodes in the 
        form [vertex ID, position vector]). 
        relevant variables:
        - <Tensor of node feature vectors >: 'feature_vector' 
        - <Tensor containing edge information (u,v)> 'adjacency_list' 
        returns:
        - <PyTorch Geometric Data Object>: 'coco_scene_graph' formed from feature_vector and adjacency_list
        '''

        '''
        Constructing node_centres. node_centres[i] returns a tuple (x,y) relating to the centre of object i 
        '''
        MH, MW = masks.size()[1:]
        node_centres = [] 
        for i, node in enumerate(nodes):
            x0, y0, x1, y1 = boxes[i]
            mask = masks[i] == 1
            xs = torch.linspace(x0, x1, MW).view(1, MW).expand(MH, MW)
            ys = torch.linspace(y0, y1, MH).view(MH, 1).expand(MH, MW)
            if mask.sum() == 0:  # Compute means of objects, if object mask sums to zero take midpoint
                mean_x = 0.5 * (x0 + x1)
                mean_y = 0.5 * (y0 + y1)
            else:
                mean_x = xs[mask].mean()
                mean_y = ys[mask].mean()
            node_centres.append([mean_x, mean_y])
        node_centres = torch.FloatTensor(node_centres)

        # Removing supernode from node_centres and nodes so that it is not considered in MST algorithm
        node_centres =  node_centres[:-1]
        nodes = nodes[:-1]

        '''
        Applying Kruskals Algorithm to generate the minimum spanning scene graph based on euclidean distance 
        '''
        # Create a new graph
        G = nx.Graph()

        # Add nodes to graph. Each node is it's integer vertex ID
        for index, _ in enumerate(nodes):
            G.add_node(index)

        num_nodes = len(nodes)
        # Create list of edges in form [u,v,euclidean distance]
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes): # to avoid duplicates
                # dist calculates the euclidean distance between two object centres
                dist = lambda u, v: math.sqrt((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2)
                euclidean_distance = dist(node_centres[i], node_centres[j])
                #edge_representation = (i, j, euclidean_distance)
                G.add_edge(i,j,distance = euclidean_distance)

        # Find the minimum spanning tree using Kruskal's algorithm. Mst is the list of edges in the minimum spanning tree
        mst = list(tree.minimum_spanning_edges(G, algorithm="kruskal", data=False))

        # Constructing adjacency list (edge list)
        simplify_edge = lambda edge: [edge[0], edge[1]]
        adjacency_list = [simplify_edge(edge) for edge in mst]

        # Constructing node feature vectors: each entry is [class ID, node centre]
        feature_vectors = []
        for i in range(num_nodes):
            new_feature_vector = [nodes[i], *node_centres[i]]
            feature_vectors.append(new_feature_vector)
        feature_vectors = torch.tensor(feature_vectors)

        '''
        Adding SuperNode as a node to feature_vector and adding the edges (connections) between it and 
        all other nodes to adjacency_list. Supernode coordinates are always (0,0,1,1), where 0,0 represents the bottom left of the image and 1,1 represents
        the top right of the image. 
        '''
        # Adding the supernode centre to node_centres
        node_centres = torch.cat((node_centres, torch.tensor([[0.5,0.5]])), dim=0)
        
        # supernode_ID denotes the unique ID for the supernode
        supernode_ID = len(nodes) 
        # supernode_ID = len(nodes) -1 
        in_image_classID = self.vocab['pred_name_to_idx']['__in_image__']

        # Define the feature vector for the supernode
        supernode_feature_vector = [in_image_classID, 0.5, 0.5]
        feature_vectors = torch.cat((feature_vectors, torch.tensor([supernode_feature_vector])), dim=0)
        
        # Create edges between the supernode and every object node (excluding predicate nodes)
        for i in range(num_nodes): 
            adjacency_list.append([supernode_ID, i])
            
        '''
        Adding Predicate Nodes: For each edge, make it a node with a nodeID. 
        '''
        nodeID = supernode_ID+1  # Starting node ID to give to predicate nodes
        midpoint = lambda p1, p2: ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        temporary = []
        for predicate in adjacency_list: # Predicate has the form [subjectID, objectID, euclideanDistance]
            print(f"PLEASE OGDOD {predicate}")
            subjectID  = predicate[0]
            objectID  = predicate[1]
            
            # Find the midpoint vector m from node_centres[u] node_centres[v]
            position = midpoint(node_centres[subjectID], node_centres[objectID])

            # Determine the class ID by determining the relationship between the subject and object
            classID = self.determine_relationship(s = subjectID, o = objectID, node_centres = node_centres)

            # Add predicate node to node feature vectors tensor 
            new_feature_vector = torch.tensor([[self.Direction[classID].value, position[0], position[1]]])
            
            feature_vectors = torch.cat([feature_vectors, new_feature_vector], dim=0)

            # Add edges from subject --> predicate and predicate --> object
            sub2pred = [subjectID, nodeID]
            pred2obj = [nodeID, objectID]
            temporary.append(sub2pred)
            temporary.append(pred2obj)
            nodeID+=1

        '''
        Writing to Pytorch geometric graph representation
        Graph Representation: A graph is represented using two primary tensors
        x: A tensor that contains the node features. Its size is [num_nodes, num_node_features].
        edge_index: A tensor of shape [2, num_edges], describing the connectivity of the graph.
        Each column represents an edge. If edge_index[:, i] = [src, dest],
        then node src is connected to node dest.
        '''
        adjacency_list_tensor = torch.tensor(temporary).t().contiguous()
        coco_scene_graph = Data(x=feature_vectors, edge_index=adjacency_list_tensor)
        print(feature_vectors)
        return coco_scene_graph

    
    def determine_relationship(self,s,o, node_centres):
        """
        determine_relationship determines the spatial relationship between subject and object, and returns
        a relationship of type Direction
        """
        sx0, sy0, sx1, sy1 = self.boxes[s] # subject corners
        ox0, oy0, ox1, oy1 = self.boxes[o] # object corners 
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
            return 'LEFT'
        if LEFT_BOUND <= theta < -math.pi / 4:
            return 'ABOVE'
        if -math.pi / 4 <= theta < math.pi / 4:
            return 'RIGHT'
        if math.pi / 4 <= theta < RIGHT_BOUND:
            return 'BELOW' 