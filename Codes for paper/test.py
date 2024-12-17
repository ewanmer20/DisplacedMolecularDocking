import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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

# Example usage
total_size = 10
clique_size = 4
erdos_renyi_prob = 0.5
adj_matrix, clique_vector = generate_adjacency_matrix_with_clique(total_size, clique_size, erdos_renyi_prob)
print("Adjacency Matrix:\n", adj_matrix)
print("Clique Vector:\n", clique_vector)

# Plot the adjacency matrix
plt.imshow(adj_matrix, cmap='Greys', interpolation='none')
plt.title('Adjacency Matrix')
plt.colorbar(label='Edge Presence')
plt.xlabel('Node Index')
plt.ylabel('Node Index')
plt.show()

# Create a graph from the adjacency matrix
G = nx.from_numpy_array(adj_matrix)

# Plot the graph
plt.figure(figsize=(8, 8))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
plt.title('Graph Representation')
plt.show()