import sys
sys.path.append(r'C:\Users\em1120\DisplacedMolecularDocking')
from DGBS_ArbitraryGraph_class import *

from thewalrus.quantum import probabilities
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
def generate_adjacency_matrix_with_clique(total_size, clique_size, erdos_renyi_prob):
    """
    Generates an adjacency matrix with an embedded clique of known size and a vector array indicating clique indices.
    
    Parameters:
    total_size (int): The total size of the adjacency matrix.
    clique_size (int): The size of the embedded clique.
    erdos_renyi_prob (float): The probability for the Erdős-Rényi graph.
    
    Returns:
    np.ndarray: The generated adjacency matrix.
    np.ndarray: A vector array indicating clique indices (1 for clique, 0 for non-clique).
    """
    # Generate an Erdős-Rényi graph
    adjacency_matrix = np.random.rand(total_size, total_size) < erdos_renyi_prob
    adjacency_matrix = np.triu(adjacency_matrix, 1)
    adjacency_matrix += adjacency_matrix.T
    
    # Embed the clique
    clique_indices = np.arange(clique_size)
    adjacency_matrix[np.ix_(clique_indices, clique_indices)] = 1
    
    # Ensure no self-loops
    np.fill_diagonal(adjacency_matrix, 0)
    
    # Convert boolean matrix to integer matrix
    adjacency_matrix = adjacency_matrix.astype(int)
    
    # Create the vector array for clique indices
    clique_vector = np.zeros(total_size, dtype=int)
    clique_vector[clique_indices] = 1
    
    return adjacency_matrix, clique_vector
def general_reduction(matrix, subgraph):
    """
    Reduces the given matrix based on the subgraph indicator array.
    
    Parameters:
    matrix (np.ndarray): The input matrix to be reduced.
    subgraph (np.ndarray): The subgraph indicator array (1 for included, 0 for excluded).
    
    Returns:
    np.ndarray: The reduced matrix.
    """
    indices = np.where(subgraph == 1)[0]
    reduced_matrix = matrix[np.ix_(indices, indices)]
    return reduced_matrix


# Example usage
total_size = 20
clique_size = 14
erdos_renyi_prob = 0.2
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
Adj_clique=general_reduction(Adj,subgraph_1)
n_subspace=Adj.shape[0]
Id = np.eye(n_subspace)
c_array=np.linspace(0,0.075,200)
gamma_array=np.linspace(0,0.5,200)
MaxCliqueProb_array=np.zeros((len(gamma_array),len(c_array)))
Norm_array=np.zeros((len(gamma_array),len(c_array)))
Loop_hafnian_array=np.zeros((len(gamma_array),len(c_array)))


# # d_alpha=(inv(Sigma_Q) @ gamma)# It could be like it was Sigma_Q before instead of inv(Sigma_Q)
# norm=np.exp(-0.5*np.dot(gamma,np.conj(Sigma_Q)@gamma))/np.sqrt(np.linalg.det(Sigma_Q))


#         print("c=",c_array[i],"gamma=",gamma_array[j],"MaxCliqueProb=",MaxCliqueProb)
for i in range(len(gamma_array)):
    for j in range(len(c_array)):
        
        rescaled_Adj=c_array[j]*Adj
        Sigma_Qinv = np.block([[Id, -np.conj(rescaled_Adj)], [-rescaled_Adj, Id]])
        Sigma_Q_tot = inv(Sigma_Qinv)
        gamma=gamma_array[i]*np.ones(2*n_subspace)
        norm=np.exp(-0.5*np.dot(gamma,np.conj(Sigma_Q_tot)@gamma))/np.sqrt(np.linalg.det(Sigma_Q_tot))
        # d_alpha=np.conjugate(Sigma_Q_tot @ gamma)
        # norm=np.exp(-0.5*np.dot(d_alpha,Sigma_Qinv@d_alpha))/np.sqrt(np.linalg.det(Sigma_Q_tot))
        Norm_array[i,j]=norm
        rescaled_clique=c_array[j]*Adj_clique
        reduced_diag=gamma_array[i]*np.ones(Adj_clique.shape[0])
        loop_hafnian_squared=np.abs(thw.loop_hafnian(rescaled_clique,reduced_diag))**2
        Loop_hafnian_array[i,j]=loop_hafnian_squared
        MaxCliqueProb=norm*loop_hafnian_squared
        
  
        MaxCliqueProb_array[i,j]=MaxCliqueProb
        # print("c=",c_array[i],"gamma=",gamma_array[j],"MaxCliqueProb=",MaxCliqueProb)


current_dir = os.path.dirname(__file__)
os.chdir(current_dir)

#  Plot the adjacency matrix
plt.imshow(Adj, cmap='Greys', interpolation='none')
plt.title('Adjacency Matrix')
plt.colorbar(label='Edge Presence')
plt.xlabel('Node Index')
plt.ylabel('Node Index')
plt.savefig(f'Adjacency_Matrix_size{total_size}_clique_size{clique_size}_erdos_prob{erdos_renyi_prob}.svg')
plt.savefig(f'Adjacency_Matrix_size{total_size}_clique_size{clique_size}_erdos_prob{erdos_renyi_prob}.png')
# Create a graph from the adjacency matrix
G = nx.from_numpy_array(Adj)

# Plot the graph
plt.figure(figsize=(8, 8))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
plt.title('Graph Representation')
plt.savefig(f'Graph{total_size}_clique_size{clique_size}_erdos_prob{erdos_renyi_prob}.svg',transparent=True)
plt.savefig(f'Graph{total_size}_clique_size{clique_size}_erdos_prob{erdos_renyi_prob}.png')
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
plt.savefig(f'Figure{total_size}_clique_size{clique_size}_erdos_prob{erdos_renyi_prob}.png')



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
