# import torch
# import torch_geometric
# import networkx as nx
# import matplotlib as plt
# g = torch.load('please_god.pt')
# g = torch_geometric.utils.to_networkx(g, to_undirected=False)
# nx.draw_networkx(g)
# plt.show()
import torch
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt  # Note the change here

g = torch.load("please_god.pt")
labeldict = {}  # what we want our feature vector to actually look like
i = 0  # i = the node ID
for node in g.x:
    labeldict[i] = node[0]
    i += 1

print(g.edge_index)
g = torch_geometric.utils.to_networkx(g, to_undirected=False)


# Check if the graph has nodes and edges
if not g.nodes():
    print("The graph has no nodes!")
if not g.edges():
    print("The graph has no edges!")

# # Draw the graph
# #plt.ioff()  # Turn off interactive mode
# nx.draw_networkx(g)
# plt.show()  # This will show the plot

# Use a different layout
pos = nx.spring_layout(g)

plt.figure(figsize=(12, 12))  # Increase the figure size for better visibility
nx.draw(g, pos, labels=labeldict, node_size=50,
        edge_color="gray", with_labels=True)
plt.show()
