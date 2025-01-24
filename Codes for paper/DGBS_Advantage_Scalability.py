import sys
import time
import pandas as pd
from datetime import datetime
# Add the Prakash folder to sys.path
# Add the Script_DGBS directory to sys.path
sys.path.append(r'C:\Users\em1120\DisplacedMolecularDocking')

# Import calc_unitary from scripts
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from Scripts_DGBS.probability_max_clique import *

current_dir = os.path.dirname(__file__)
plot_dir = os.path.join(current_dir, 'Plots')
os.makedirs(plot_dir, exist_ok=True)
data_dir = os.path.join(current_dir, 'Data')
os.makedirs(data_dir, exist_ok=True)
os.chdir(plot_dir)

# Initialize an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['M', 'Id', 'Max_Advantage', 'Gamma_value','Mean_disp', 'Mean_sqz','Adjacency_Matrix'])


M=np.linspace(14,30,17).astype(int)
print(M)
number_of_cliques=200
clique_size = 12
c_truncation_factor=0.5 # Truncation factor for the photon number
erdos_renyi_prob = 0.2
start_time_init = time.time() 
for i in range(len(M)):
    for j in range(number_of_cliques):
        start_time = time.time()  # Start time of the iteration
        adj_matrix, clique_vector = generate_adjacency_matrix_with_clique(M[i], clique_size, erdos_renyi_prob)
        subgraph_1=np.array(clique_vector)
        Adj=adj_matrix
        c_max=find_max_c(Adj)
        c_array=np.linspace(0,c_max*c_truncation_factor,10)
        gamma_array=np.linspace(0,5,50)
        MaxCliqueProb_array=np.zeros((len(gamma_array),len(c_array)))
        for k in range(len(gamma_array)):
            for l in range(len(c_array)):
                MaxCliqueProb_array[k,l]=probability_DGBS_subgraph(c_array[l],gamma_array[k],Adj,subgraph_1)

        max_advantage=np.max(MaxCliqueProb_array)
        gamma_max_index,c_max_index=find_max_indices(MaxCliqueProb_array)
        print(max_advantage / MaxCliqueProb_array[0, c_max_index])
            # Create a DataFrame for the current iteration
        iteration_df = pd.DataFrame([{
            'M': M[i],
            'Id': j,
            'Max_Advantage': max_advantage / MaxCliqueProb_array[0, c_max_index],
            'Gamma_value': gamma_array[gamma_max_index],
            'Mean_disp':disp_photon_number(c_array[c_max_index],gamma_array[gamma_max_index],adj_matrix),
            'Mean_sqz': sqz_photon_number(c_array[c_max_index],adj_matrix),
            'Adjacency_Matrix': np.ravel(adj_matrix)
        }])

        # Concatenate the current iteration DataFrame with the results DataFrame
        results_df = pd.concat([results_df, iteration_df], ignore_index=True)
        end_time = time.time()  # End time of the iteration
        duration = end_time - start_time  # Duration of the iteration
        print(f"Iteration (i={i}, j={j}) took {duration:.2f} seconds")
end_time = time.time()  # End time of the iterations
print(f"Total time taken: {end_time - start_time_init:.2f} seconds")
now = datetime.now()
formatted_time = now.strftime("%Y%m%d_%H%M%S")
# Save the DataFrame to a CSV file in the Data subdirectory
filename = f"results_{formatted_time}.csv"
results_df.to_csv(os.path.join(data_dir, filename), index=False)

# Calculate the average Max_Advantage grouped by M
average_max_advantage = results_df.groupby('M')['Max_Advantage'].mean()
variance_max_advantage = results_df.groupby('M')['Max_Advantage'].var()
average_max_sqz = results_df.groupby('M')['Mean_sqz'].mean()
variance_max_sqz = results_df.groupby('M')['Mean_sqz'].var()
average_gamma_value = results_df.groupby('M')['Gamma_value'].mean()
variance_gamma_value = results_df.groupby('M')['Gamma_value'].var()
average_mean_disp = results_df.groupby('M')['Mean_disp'].mean()
# Plot the result
fig, ax = plt.subplots(1, 2, figsize=(18, 6))

ax[0].errorbar(average_max_advantage.index, average_max_advantage.values, yerr=variance_max_advantage.values, fmt='o', linestyle='-', color='b', ecolor='b', capsize=5)
ax[0].errorbar(average_gamma_value.index, average_gamma_value.values, yerr=variance_gamma_value.values, fmt='o', linestyle='-', color='r', ecolor='r', capsize=5)
ax[1].plot(average_mean_disp.index, average_mean_disp.values, linestyle='-', color='y',marker='o',label='Mean Displacement')
ax[1].plot(average_mean_disp.index,average_mean_disp.values/average_max_sqz.values, marker='o',color='r', linestyle='-',label='Ratio Disp/Sqz')
ax[1].errorbar(average_max_sqz.index, average_max_sqz.values, yerr=3*variance_max_sqz.values, fmt='o', linestyle='-', color='g', ecolor='g', capsize=5,label='Mean Squeezing')
ax[0].legend(['Max Advantage','Gamma Value'])
ax[1].legend()
ax[0].set_xlabel('M')
ax[1].set_xlabel('M')
ax[0].set_ylabel('Advantage')
ax[1].set_ylabel('Values')
ax[0].set_title(f"Average Max Advantage and Gamma Value as a function of M for c={c_truncation_factor:.2f}*c_max and {number_of_cliques} cliques")
ax[1].set_title(f"Optimal Mean Squeezing and Displacement as a function of M for c={c_truncation_factor:.2f}*c_max and {number_of_cliques} cliques")
ax[0].grid(True)
ax[1].grid(True)
ax[0].set_xticks(M)
ax[1].set_xticks(M)
plt.savefig(os.path.join(plot_dir, f"average_max_advantage_{formatted_time}.svg"))
plt.show()