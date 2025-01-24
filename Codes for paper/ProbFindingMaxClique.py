
import sys
# Add the Prakash folder to sys.path
# Add the Script_DGBS directory to sys.path
sys.path.append(r'C:\Users\em1120\DisplacedMolecularDocking')


import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from Scripts_DGBS.DGBS_ArbitraryGraph_class import *
from Scripts_DGBS.probability_max_clique import generate_adjacency_matrix_with_clique, general_reduction,probability_DGBS_subgraph


current_dir = os.path.dirname(__file__)
plot_dir = os.path.join(current_dir, 'Plots')
os.makedirs(plot_dir, exist_ok=True)
os.chdir(plot_dir)


# Example usage
total_size = 20
clique_size = 14
erdos_renyi_prob = 0.2
# adj_matrix, clique_vector = generate_adjacency_matrix_with_clique(total_size, clique_size, erdos_renyi_prob)
# print("Adjacency Matrix:\n", adj_matrix)
# print("Clique Vector:\n", clique_vector)
hbar=2
Adj=np.array([
  [0, 1, 1, 1, 0, 0],
  [1, 0, 1, 1, 0, 1],
  [1, 1, 0, 1, 0, 0],
  [1, 1, 1, 0, 1, 0],
  [0, 0, 0, 1, 0, 1],
  [0, 1, 0, 0, 1, 0]
])
subgraph_1=np.array([1,1,1,1,0,0])
# Adj=adj_matrix
# subgraph_1=np.array(clique_vector)
print("Adjacency Matrix:\n", Adj)
print("Clique Vector:\n", subgraph_1)
n_subspace=Adj.shape[0]
Id = np.eye(n_subspace)
c_array=np.linspace(0,0.3,100)
gamma_array=np.linspace(0,1.5,100)
MaxCliqueProb_array=np.zeros((len(gamma_array),len(c_array)))
Norm_array=np.zeros((len(gamma_array),len(c_array)))
Loop_hafnian_array=np.zeros((len(gamma_array),len(c_array)))
for i in range(len(gamma_array)):
    for j in range(len(c_array)):
        MaxCliqueProb_array[i,j]=probability_DGBS_subgraph(c_array[j],gamma_array[i],Adj,subgraph_1)

#  Plot the adjacency matrix
plt.imshow(Adj, cmap='Greys', interpolation='none')
plt.title('Adjacency Matrix')
plt.colorbar(label='Edge Presence')
plt.xlabel('Node Index')
plt.ylabel('Node Index')
plt.savefig(f'Adjacency_Matrix_size{total_size}_clique_size{clique_size}_erdos_prob{erdos_renyi_prob}.svg')
# Create a graph from the adjacency matrix
G = nx.from_numpy_array(Adj)

# Plot the graph
plt.figure(figsize=(8, 8))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
plt.title('Graph Representation')
plt.savefig(f'Graph{total_size}_clique_size{clique_size}_erdos_prob{erdos_renyi_prob}.svg',transparent=True)
plt.rcParams.update({'font.size': 40})
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(24, 24))
im=ax.imshow(MaxCliqueProb_array,origin='lower')

plt.colorbar(im,fraction=0.05)
ax.set_title('Probability Max Clique')
contour = ax.contour(MaxCliqueProb_array, levels=8, colors='white',origin='lower')
ax.set_xlabel('c') 
ax.set_ylabel('gamma') 
ax.clabel(contour, inline=True, fontsize=8, colors='white')
# Set the tick positions and labels for the original matrix
ax.set_xticks(np.linspace(0, len(c_array) - 1, 5))
ax.set_xticklabels(np.round(np.linspace(c_array[0], c_array[-1], 5), 2))
ax.set_yticks(np.linspace(0, len(gamma_array) - 1, 5))
ax.set_yticklabels(np.round(np.linspace(gamma_array[0], gamma_array[-1], 5), 2))
plt.savefig(f'Figure{total_size}_clique_size{clique_size}_erdos_prob{erdos_renyi_prob}.svg')

# im1=ax[1].imshow(Norm_array,origin='lower')
# ax[1].set_title('Normalisation')
# # divider1 = make_axes_locatable(ax[1])
# # cax1 = divider1.append_axes("right", size="5%",pad="5%")
# # cbar1 = fig.colorbar(im1, ax=ax[1])
# contour1 = ax[1].contour(Norm_array, levels=8, colors='white',origin='lower')
# ax[1].clabel(contour1, inline=True, fontsize=8, colors='white')
# # Set the tick positions and labels for the original matrix
# ax[1].set_xticks(np.linspace(0, len(c_array) - 1, 5))
# ax[1].set_xticklabels(np.round(np.linspace(c_array[0], c_array[-1], 5), 2))
# ax[1].set_yticks(np.linspace(0, len(gamma_array) - 1, 5))
# # Add a color bar
# ax[1].set_yticklabels(np.round(np.linspace(gamma_array[0], gamma_array[-1], 5), 2))
# im2=ax[2].imshow(Loop_hafnian_array,origin='lower')
# # divider2 = make_axes_locatable(ax[2])
# # cax2 = divider2.append_axes("right", size="5%",pad="5%")
# # cbar2 = fig.colorbar(im2, ax=ax[2])

# ax[2].set_title('|LoopHafnian|^2')
# contour1 = ax[2].contour(Loop_hafnian_array, levels=8, colors='white',origin='lower')
# ax[2].clabel(contour1, inline=True, fontsize=8, colors='white')
# # Set the tick positions and labels for the original matrix
# ax[2].set_xticks(np.linspace(0, len(c_array) - 1, 5))
# ax[2].set_xticklabels(np.round(np.linspace(c_array[0], c_array[-1], 5), 2))
# ax[2].set_yticks(np.linspace(0, len(gamma_array) - 1, 5))
# ax[2].set_yticklabels(np.round(np.linspace(gamma_array[0], gamma_array[-1], 5), 2))
plt.show()

