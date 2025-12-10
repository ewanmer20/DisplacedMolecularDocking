import numpy as np
from numpy.linalg import inv
import thewalrus as thw
from thewalrus.quantum import probabilities
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.special import factorial
import itertools
import strawberryfields as sf
from strawberryfields.ops import Interferometer, Sgate, Dgate, LossChannel
from strawberryfields.decompositions import takagi



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
    subgraph (np.ndarray): The subgraph indicator array (1 for included, >1 for repeated).
    
    Returns:
    np.ndarray: The reduced matrix.
    """
    indices = np.where(subgraph >= 1)[0]
    repeated_indices = []
    
    for idx in indices:
        repeated_indices.extend([idx] * int(subgraph[idx]))
    
    reduced_matrix = matrix[np.ix_(repeated_indices, repeated_indices)]
    return reduced_matrix

# def probability_max_clique_DGBS(c, gamma_val, Adj, Adj_clique):
#     """
#     Calculates the probability of finding the maximum clique.

#     Parameters:
#     c (float): The constant c.
#     gamma_val (float): The loop strength.
#     Adj (np.ndarray): The adjacency matrix.
#     Adj_clique (np.ndarray): The reduced adjacency matrix.

#     Returns:
#     float: The probability of finding the maximum clique.

#     """
#     Id = np.eye(Adj.shape[0])
#     rescaled_Adj=c*Adj
#     Sigma_Qinv = np.block([[Id, -np.conj(rescaled_Adj)], [-rescaled_Adj, Id]])
#     Sigma_Q_tot = inv(Sigma_Qinv)
#     gamma=gamma_val*np.ones(2*Adj.shape[0])
#     norm=np.exp(-0.5*np.dot(gamma,np.conj(Sigma_Q_tot)@gamma))/np.sqrt(np.linalg.det(Sigma_Q_tot))
#     # d_alpha=np.conjugate(Sigma_Q_tot @ gamma)
#     # norm=np.exp(-0.5*np.dot(d_alpha,Sigma_Qinv@d_alpha))/np.sqrt(np.linalg.det(Sigma_Q_tot))
#     rescaled_clique=c*Adj_clique
#     reduced_diag=gamma_val*np.ones(Adj_clique.shape[0])
#     loop_hafnian_squared=np.abs(thw.loop_hafnian(rescaled_clique,reduced_diag))**2
#     MaxCliqueProb=norm*loop_hafnian_squared
#     return MaxCliqueProb

def find_max_c(Adj):
    """
    Finds the maximal value of c such that all the eigenvalues of c * Adj are between 0 and 1.

    Parameters:
    Adj (np.ndarray): The adjacency matrix.

    Returns:
    float: The maximal value of c.
    """
    # Compute the takagi of the adjacency matrix
    (rl,_)=takagi(Adj)

    
    # Find the maximum takagi eigenvalue
    max_rl = np.max(np.abs(rl))
    
    # The maximal value of c
    max_c = 1 / max_rl
    
    return max_c

def find_max_indices(A):
    """
    Finds the x and y indices of the maximal element in a 2D array A.

    Parameters:
    A (np.ndarray): The input 2D array.

    Returns:
    tuple: The (x, y) indices of the maximal element.
    """
    max_index = np.argmax(A)
    max_indices = np.unravel_index(max_index, A.shape)
    return max_indices

def probability_DGBS_subgraph(c, gamma_val, Adj, subgraph):
    """
    Calculates the probability of finding a subgraph.
    
    Parameters:
    c (float): The constant c.
    gamma_val (float): The loop strength.
    Adj (np.ndarray): The adjacency matrix.
    subgraph (np.ndarray): The subgraph indicator array (1 for included, 0 for excluded, if the integer is greater than one, it will repeat the rows and the columns).
    
    Returns:
    float: The probability of finding the subgraph.
    """
    Id = np.eye(Adj.shape[0])
    rescaled_Adj = c * Adj
    Sigma_Qinv = np.block([[Id, -np.conj(rescaled_Adj)], [-rescaled_Adj, Id]]
    )
    Sigma_Q_tot = inv(Sigma_Qinv)
    gamma = gamma_val * np.ones(2 * Adj.shape[0])
    if np.linalg.det(Sigma_Q_tot)<= 0:
        print("Determinant of Sigma_Q_tot is negative")
        return -1
    norm = np.exp(-0.5 * np.dot(gamma, np.conj(Sigma_Q_tot) @ gamma)) / (np.sqrt(np.linalg.det(Sigma_Q_tot))*np.prod(factorial(subgraph)))
    rescaled_subgraph = c * general_reduction(Adj, subgraph)
    reduced_diag = gamma_val * np.ones(rescaled_subgraph.shape[0])
    loop_hafnian_squared = np.abs(thw.loop_hafnian(rescaled_subgraph, reduced_diag)) ** 2
    subgraph_prob = norm * loop_hafnian_squared
    return subgraph_prob

def probability_array_DGBS(c,gamma_val,Adj,loss,cutoff=5,fock_prob=True):
    """Return the tensor of the probabilities for DGBSwith strawberryfields.

    Parameters:
    c (float): The constant c.
    gamma_val (float): The loop strength.
    Adj (np.ndarray): The adjacency matrix.
    cutoff (int): The cutoff for the dimension of Fock space for each mode.
    """
    m=Adj.shape[0]
    Id = np.eye(Adj.shape[0])
    rescaled_Adj = c * Adj
    gamma = gamma_val * np.ones(2*Adj.shape[0])
    Sigma_Qinv = np.block([[Id, -np.conj(rescaled_Adj)], [-rescaled_Adj, Id]])
    Sigma_Q_tot = inv(Sigma_Qinv)
    gamma = gamma_val * np.ones(2 * Adj.shape[0])
    d_alpha = (Sigma_Q_tot @ gamma)
    alpha = d_alpha[:m]

    rl, U = sf.decompositions.takagi(rescaled_Adj)
    # create the m mode Strawberry Fields program
    gbs = sf.Program(m)


    r = np.arctanh(rl)

    with gbs.context as q:
    # prepare the input squeezed states
      for n in range(m):
          Sgate(-r[n]) | q[n]

    # linear interferometer
      Interferometer(U) | q

    #Displacement operation    
      for n in range(m):
        if alpha[n] < 0:
            phase = np.pi
        else:
            phase = 0
        Dgate(np.abs(alpha[n]), phase) | q[n]
      for n in range(m):
          LossChannel(1.0 - loss) | q[n]

   


    eng = sf.Engine(backend="gaussian")
    results = eng.run(gbs)
    state = results.state
    if fock_prob==True:
        return state.all_fock_probs(cutoff=cutoff)
    else:
        return state


def generate_lists(m, n_cutoff):
    """
    Generates all lists of length m where the elements are less than or equal to n_cutoff.

    Parameters:
    m (int): The length of the lists.
    n_cutoff (int): The maximum value for the elements in the lists.

    Returns:
    list: A list of lists satisfying the condition.
    """
    return list(itertools.product(range(n_cutoff), repeat=m))

def probability_lossy_DGBS(c,gamma_val,loss,Adj,prob_array,subgraph,cutoff=5):
    """
    Calculates the probability of finding a subgraph given  loss in the system modeled as detection loss.

    Parameters:
    c (float): The constant c.
    gamma_val (float): The loop strength.
    loss(float): The uniform loss in the system.
    Adj (np.ndarray): The adjacency matrix.
    subgraph (np.ndarray): The subgraph indicator array (1 for included, 0 for excluded, if the integer is greater than one, it will repeat the rows and the columns).
    cutoff(int): The cutoff for the dimension of Fock space for each mode.
    
    Returns:
    total_prob: The probability of finding the subgraph given the loss
    prob_array: The probability array for the subgraph without loss
      """
    if prob_array is None:
        prob_array=probability_array_DGBS(c, gamma_val, Adj, cutoff=cutoff)

    m=Adj.shape[0]
    indices_ensemble=generate_lists(m,cutoff)
    total_prob=0
    for indices_list in indices_ensemble:
        weigth=1
        for k in range(len(indices_list)):
            if subgraph[k]==0:
                weigth*=(loss)**(indices_list[k])
            elif subgraph[k]>0:
                if indices_list[k]<subgraph[k]:
                    weigth*=0
                else:    
                    weigth*=1-(loss)**(indices_list[k])
        total_prob+=weigth*prob_array[indices_list]
    return total_prob,prob_array
  
def sqz_photon_number(c,Adj):
    """
    Returns the squeezed photon number for the DGBS model.

    Parameters:
    c (float): The constant c.
    Adj (np.ndarray): The adjacency matrix.

    Returns:
    float: The squeezed photon number.
    """
    rl=takagi(c*Adj)[0]
    return np.sum(np.abs(np.sinh(np.arctanh(rl)))**2)

def disp_photon_number(c,gamma_val,Adj):
    """
    Returns the displaced photon number for the DGBS model.

    Parameters:
    c (float): The constant c.
    gamma_val (float): The loop strength.
    Adj (np.ndarray): The adjacency matrix.

    Returns:
    float: The displaced photon number.
    """
    m=Adj.shape[0]
    Id = np.eye(Adj.shape[0])
    rescaled_Adj = c * Adj
    gamma = gamma_val * np.ones(2*Adj.shape[0])
    Sigma_Qinv = np.block([[Id, -np.conj(rescaled_Adj)], [-rescaled_Adj, Id]])
    Sigma_Q_tot = inv(Sigma_Qinv)
    gamma = gamma_val * np.ones(2 * Adj.shape[0])
    d_alpha = (Sigma_Q_tot @ gamma)
    alpha = d_alpha[:m]
    return np.sum(np.abs(alpha)**2)

if __name__=="__main__":
    matrix=np.array([[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14],[15,16,17,18,19],[20,21,22,23,24]])
    subgraph=np.array([1,0,2,0,0])
    print(matrix)
    print(general_reduction(matrix, subgraph))
    print(probability_DGBS_subgraph(0.01, 0.2, matrix, subgraph))
    m = 3
    n_cutoff = 2
    all_lists = generate_lists(m, n_cutoff)
    print(all_lists)
    matrix=np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]])
    print(disp_photon_number(0.01,0.2,matrix))
    prob=probability_array_DGBS(0.01, 0.2,matrix, cutoff=5)
    print(prob)
    print(prob.shape)
    print(prob[tuple([1,1,1])])
    lossy_prob,_=probability_lossy_DGBS(c=0.1, gamma_val=0.5, loss=0.1, Adj=matrix,prob_array=None,subgraph=np.array([0,0,0,0]), cutoff=5)
    print(f"{lossy_prob:.2e}")
    # Example usage
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 9, 8]])
    x, y = find_max_indices(A)
    print(f"The maximal element is at indices: ({x}, {y})")

