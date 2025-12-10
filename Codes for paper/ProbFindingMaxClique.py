
import sys
# Add the Prakash folder to sys.path
# Add the Script_DGBS directory to sys.path
sys.path.append(r'C:\Users\em1120\DisplacedMolecularDocking')


import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from strawberryfields.decompositions import takagi
from Scripts_DGBS.DGBS_ArbitraryGraph_class import *
from Scripts_DGBS.probability_max_clique import generate_adjacency_matrix_with_clique, general_reduction,probability_DGBS_subgraph
import matplotlib.ticker as ticker

current_dir = os.path.dirname(__file__)
plot_dir = os.path.join(current_dir, 'Plots')
os.makedirs(plot_dir, exist_ok=True)
os.chdir(plot_dir)


# Example usage
total_size = 24
clique_size = 4
erdos_renyi_prob = 0.4
adj_matrix, clique_vector = generate_adjacency_matrix_with_clique(total_size, clique_size, erdos_renyi_prob)
print("Adjacency Matrix:\n", adj_matrix)
print("Clique Vector:\n", clique_vector)
hbar=2
# Adj=np.array([
#   [0, 1, 1, 1, 0, 0],
#   [1, 0, 1, 1, 0, 1],
#   [1, 1, 0, 1, 0, 0],
#   [1, 1, 1, 0, 1, 0],
#   [0, 0, 0, 1, 0, 1],
#   [0, 1, 0, 0, 1, 0]
# ])
# subgraph_1=np.array([1,1,1,1,0,0])
Adj=adj_matrix
subgraph_1=np.array(clique_vector)
print("Adjacency Matrix:\n", Adj)
print("Clique Vector:\n", subgraph_1)
n_subspace=Adj.shape[0]
Id = np.eye(n_subspace)
c_array=np.linspace(0.01,1/np.max(takagi(Adj)[0])*0.95,200)
gamma_array=np.linspace(0,1,200)
tanh_max_array=np.array([np.max(takagi(c*Adj)[0]) for c in c_array])
MaxCliqueProb_array=np.zeros((len(gamma_array),len(c_array)))
Norm_array=np.zeros((len(gamma_array),len(c_array)))
Loop_hafnian_array=np.zeros((len(gamma_array),len(c_array)))
for i in range(len(gamma_array)):
    for j in range(len(c_array)):
        MaxCliqueProb_array[i,j]=probability_DGBS_subgraph(c_array[j],gamma_array[i],Adj,subgraph_1)
Ref_array=MaxCliqueProb_array[0,:]
Improvement_array=[MaxCliqueProb_array[i,j]/Ref_array[j] for i in range(len(gamma_array)) for j in range(len(c_array))]
Improvement_array=np.array(Improvement_array).reshape(len(gamma_array),len(c_array))
MaxImprovement = np.max(Improvement_array, axis=0)
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
plt.rcParams.update({'font.size': 48})
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(24, 24))
im=ax.imshow(MaxCliqueProb_array,origin='lower')

plt.colorbar(im,fraction=0.05)
contour = ax.contour(MaxCliqueProb_array, levels=8, colors='white',origin='lower')
ax.set_xlabel(r'$\lambda_{max}$') 
ax.set_ylabel(r'$\gamma$') 
ax.clabel(contour, inline=True, fontsize=8, colors='white')
# Set the tick positions and labels for the original matrix
ax.set_xticks(np.linspace(0, len(tanh_max_array) - 1, 5))
ax.set_xticklabels(np.round(np.linspace(tanh_max_array[0], tanh_max_array[-1], 5), 2))
ax.set_yticks(np.linspace(0, len(gamma_array) - 1, 5))
ax.set_yticklabels(np.round(np.linspace(gamma_array[0], gamma_array[-1], 5), 2))
plt.savefig(f'Figure{total_size}_clique_size{clique_size}_erdos_prob{erdos_renyi_prob}.svg')


# fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(24, 24))
# im2=ax2.imshow(Improvement_array,origin='lower')
# plt.colorbar(im2,fraction=0.05)
# contour = ax2.contour(Improvement_array, levels=8, colors='white',origin='lower')
# ax2.set_xlabel(r'$\lambda_{max}$') 
# ax2.set_ylabel(r'$\gamma$') 
# ax2.clabel(contour, inline=True, fontsize=8, colors='white')
# # Set the tick positions and labels for the original matrix
# ax2.set_xticks(np.linspace(0, len(tanh_max_array) - 1, 5))
# ax2.set_xticklabels(np.round(np.linspace(tanh_max_array[0], tanh_max_array[-1], 5), 2))
# ax2.set_yticks(np.linspace(0, len(gamma_array) - 1, 5))
# ax2.set_yticklabels(np.round(np.linspace(gamma_array[0], gamma_array[-1], 5), 2))
# # Find the maximum improvement for each value of c

# print("Max Improvement Array:\n", MaxImprovement)
# Plot the maximum improvement as a function of the squeezing parameter (c)
plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 30})
plt.plot(tanh_max_array[1:], MaxImprovement[1:], marker='o', linestyle='-', color='blue', label='Max Improvement (DGBS/GBS)')
plt.xlabel(r'Squeezing Parameter ($\lambda_{max}$)', fontsize=14)
plt.ylabel('Max Improvement', fontsize=14)
plt.yscale('log')
plt.title('Maximum Improvement Between GBS and DGBS as a Function of Squeezing', fontsize=16)
# Add smaller ticks on the log scale
ax = plt.gca()
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())  # Optionally hide minor tick labels

plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid for both major and minor ticks
plt.legend(fontsize=12)
plt.savefig(f'MaxImprovement_vs_Squeezing_{total_size}_clique_size{clique_size}_erdos_prob{erdos_renyi_prob}.svg')
plt.show()
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


