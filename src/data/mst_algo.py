import networkx as nx
from networkx.algorithms import tree
## imports for set operations on kruskals


## computing the centre of all objects
obj_centers = []
_, MH, MW = masks.size()
for i, obj_idx in enumerate(objs):
    x0, y0, x1, y1 = boxes[i]
    mask = (masks[i] == 1) ### only for objects
    xs = torch.linspace(x0, x1, MW).view(1, MW).expand(MH, MW)
    ys = torch.linspace(y0, y1, MH).view(MH, 1).expand(MH, MW)
    if mask.sum() == 0: ### compute means of objects, if for some reason object mask sums to zero just take midpoint
        mean_x = 0.5 * (x0 + x1)
        mean_y = 0.5 * (y0 + y1)
    else:
        mean_x = xs[mask].mean()
        mean_y = ys[mask].mean()
    obj_centers.append([mean_x, mean_y])
obj_centers = torch.FloatTensor(obj_centers)


## applying kruskals algorithm to generate the minimum spanning 
## scene graph based on euclidean distance 


# Create a new graph
G = nx.Graph()

# Add nodes to graph. Each node is it's integer vertex ID
for index, currentObject in enumerate(objs):
    # keep as enumerate as not sure if we want these properties later
    # x = obj_centers[currentObject][0]
    # y = obj_centers[currentObject]
    # node_representation = (index, (x,y))
    G.add_node(index)

num_nodes = len(objs)
# Create list of edges in form [u,v,euclidean distance]
for i in range(num_nodes):#for each node
    for j in range(i+1, num_nodes): # to avoid duplicate edges
        #TODO: Import math
        # dist calculates the euclidean distance between two object centres
        dist = (lambda u, v :sqrt((u[0]-v[0])^2+(u[1]-v[1])^2)) 
        edge_representation = (i,j,dist(obj_centres(i), obj_centres(j))) 
        G.add_edge(edge_representation)


# Find the minimum spanning tree using Kruskal's algorithm
mst = tree.minimum_spanning_edges(G, algorithm="kruskal", data=False)
edgelist = list(mst) #### this is our triples





