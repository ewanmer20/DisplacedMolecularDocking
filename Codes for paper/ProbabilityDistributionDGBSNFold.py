import sys
import time
import itertools
# Add the Prakash folder to sys.path
# Add the Script_DGBS directory to sys.path
sys.path.append(r'C:\Users\em1120\DisplacedMolecularDocking')
from datetime import datetime
# Import calc_unitary from scripts
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import comb
from scipy.stats import entropy
from Scripts_DGBS.probability_max_clique import *

plt.rcParams.update({'font.size': 32})


def generate_combinations(n, k):
    """
    Generate all possible combinations of placing k ones in a list of n zeros.

    Parameters:
    n (int): Length of the list.
    k (int): Number of ones to place in the list.

    Returns:
    list: A list of lists, each containing a combination of k ones in a list of n zeros.
    """
    # Generate all possible positions to place the ones
    positions = itertools.combinations(range(n), k)
    
    # Create the list of combinations
    combinations = []
    for pos in positions:
        combination = [0] * n
        for p in pos:
            combination[p] = 1
        combinations.append(combination)
    
    return combinations

current_dir = os.path.dirname(__file__)
plot_dir = os.path.join(current_dir, 'Plots')
os.makedirs(plot_dir, exist_ok=True)
data_dir = os.path.join(current_dir, 'Data')
os.makedirs(data_dir, exist_ok=True)
os.chdir(plot_dir)


gamma_val=[0,0.2,1]
cutoff=2
M=18 # Total number of nodes
clique_size=6
erdos_renyi_prob=0.2
results_df = pd.DataFrame(columns=['c', 'Gamma_value','Mean_photon_sqz','Mean_photon_disp','Adjacency_Matrix','Distribution'])
loss_result=[]
start=time.time()

adj_matrix, clique_vector = generate_adjacency_matrix_with_clique(M, clique_size, erdos_renyi_prob)
c_max=find_max_c(adj_matrix)
c_trunc=0.5
for j in range(len(gamma_val)):
    start_i=time.time()
    state=probability_array_DGBS(c_max*c_trunc, gamma_val[j], adj_matrix,loss=0, cutoff=cutoff,fock_prob=False)
    prob_maxclique=state.fock_prob(tuple(clique_vector),cutoff=20)

    # Fock states to measure at output

    basis = generate_combinations(M, clique_size)

    # extract the probabilities of calculating several
    # different Fock states at the output, and print them to the terminal

    nn = 0
    distribution = np.zeros(int(comb(M,clique_size)))
    print("Probability of the maximum clique:", prob_maxclique)
    print("Distribution length", len(distribution))
    for combo in basis:
        prob = state.fock_prob(list(combo), cutoff=20)
        distribution[nn] = prob
        nn += 1
    distribution_sorted = np.sort(distribution)
    distribution_desc = distribution_sorted[::-1]
    serie_df = pd.DataFrame([{
        'c': c_max*c_trunc,
        'Gamma_value': gamma_val[j],
        'Mean_photon_sqz':sqz_photon_number(c_max*c_trunc,adj_matrix),
        'Mean_photon_disp':disp_photon_number(c_max*c_trunc,gamma_val[j],adj_matrix),
        'Adjacency_Matrix': np.ravel(adj_matrix),
        'Distribution': distribution_desc
    }])

    results_df = pd.concat([results_df, serie_df], ignore_index=True)
    end_i=time.time()
    print("Time taken for compute probability distribution:", end_i-start_i)
    end=time.time()
    print("Time taken for compute probability distribution:", end-start)

now = datetime.now()
formatted_time = now.strftime("%Y%m%d_%H%M%S")
# Save the DataFrame to a CSV file in the Data subdirectory
filename = f"results_{formatted_time}.csv"

results_df.to_csv(os.path.join(data_dir, filename), index=False)

# Plot the probability distribution
fig, ax = plt.subplots(figsize=(10, 6))
for gamma in results_df['Gamma_value'].unique():
    subset = results_df[results_df['Gamma_value'] == gamma]
    normalized_distribution=subset['Distribution'].values[0]/np.sum(subset['Distribution'].values[0])
    shannon_entropy=entropy(normalized_distribution)
    if gamma==0:
        entrop_ref=shannon_entropy
    
    ax.plot(np.arange(int(comb(M, clique_size))), normalized_distribution, marker='o', linestyle='-', label=fr'$\gamma = {gamma:.2f}$, $H_s$ = {shannon_entropy/entrop_ref:.2f}')
    
entropy_uniform=entropy(np.full(int(comb(M, clique_size)), 1/int(comb(M, clique_size))))
ax.plot(np.arange(int(comb(M, clique_size))), np.full(int(comb(M, clique_size)), 1/int(comb(M, clique_size))), marker='x', linestyle='--', color='red', label=f'Uniform Distribution, $H_s$ = {entropy_uniform/entrop_ref:.2f}')
ax.set_yscale('log')
ax.set_xlabel('Instances')
ax.set_ylabel('Probability')
ax.set_title(f'Probability distribution for {clique_size} clique size, {M} nodes, and {erdos_renyi_prob} edge probability')
ax.legend(loc='upper right')

# Show the figure
plt.show()


    