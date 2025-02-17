import sys
import time
import pandas as pd
from datetime import datetime
from strawberryfields.decompositions import takagi
# Add the Prakash folder to sys.path
# Add the Script_DGBS directory to sys.path
sys.path.append(r'C:\Users\em1120\DisplacedMolecularDocking')

# Import calc_unitary from scripts
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from Scripts_DGBS.probability_max_clique import *
from scipy.optimize import curve_fit

current_dir = os.path.dirname(__file__)
plot_dir = os.path.join(current_dir, 'Plots')
os.makedirs(plot_dir, exist_ok=True)
data_dir = os.path.join(current_dir, 'Data')
os.makedirs(data_dir, exist_ok=True)
os.chdir(plot_dir)

# Initialize an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['M', 'Id', 'Max_Advantage', 'Gamma_value','Mean_disp', 'Mean_sqz','Adjacency_Matrix'])

clique_size = 20 # Size of the cliques
offset_M=2 # Offset for the M values
c_truncation_factor=0.5 # Truncation factor for the photon number
erdos_renyi_prob = 0.8 # Probability of an edge in the Erdos-Renyi graph
number_of_M=14 # Total number of M values to generate
number_of_cliques=1 # Number of cliques to generate per M 
gamma_truncation_factor=1 # Truncation factor for the gamma value
gamma_number=100 # Number of gamma values to generate 
M=np.arange(clique_size+offset_M, clique_size+number_of_M+offset_M-1, 2).astype(int)
start_time_init = time.time() 
for i in range(len(M)):
    for j in range(number_of_cliques):
        start_time = time.time()  # Start time of the iteration
        adj_matrix, clique_vector = generate_adjacency_matrix_with_clique(M[i], clique_size, erdos_renyi_prob)
        subgraph_1=np.array(clique_vector)
        Adj=adj_matrix
        c_max=find_max_c(Adj)
        (rl,_)=takagi(0.8*c_max*adj_matrix)
        print(np.sinh(np.arctanh(rl))**2)
        c_array=np.linspace(0,c_max*c_truncation_factor,10)
        gamma_array=np.linspace(0,gamma_truncation_factor,gamma_number)
        MaxCliqueProb_array=np.zeros(len(gamma_array))
        for k in range(len(gamma_array)):
            MaxCliqueProb_array[k]=probability_DGBS_subgraph(c_max*c_truncation_factor,gamma_array[k],Adj,subgraph_1)
        gamma_max_index=np.argmax(MaxCliqueProb_array)
        max_advantage=np.max(MaxCliqueProb_array)
        # print(max_advantage / MaxCliqueProb_array[0])
        iteration_df = pd.DataFrame([{
            'M': M[i],
            'Id': j,
            'Max_Advantage': max_advantage / MaxCliqueProb_array[0],
            'Gamma_value': gamma_array[gamma_max_index],
            'Mean_disp':disp_photon_number(c_max*c_truncation_factor,gamma_array[gamma_max_index],adj_matrix),
            'Mean_sqz': sqz_photon_number(c_max*c_truncation_factor,adj_matrix),
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
# formatted_time = now.strftime("%Y%m%d_%H%M%S")
# # Save the DataFrame to a CSV file in the Data subdirectory
# filename = f"results_{formatted_time}.csv"
# results_df.to_csv(os.path.join(data_dir, filename), index=False)

# # Calculate the average Max_Advantage grouped by M
# average_max_advantage = results_df.groupby('M')['Max_Advantage'].mean()
# variance_max_advantage = results_df.groupby('M')['Max_Advantage'].var()
# average_max_sqz = results_df.groupby('M')['Mean_sqz'].mean()
# variance_max_sqz = results_df.groupby('M')['Mean_sqz'].var()
# average_gamma_value = results_df.groupby('M')['Gamma_value'].mean()
# variance_gamma_value = results_df.groupby('M')['Gamma_value'].var()
# average_mean_disp = results_df.groupby('M')['Mean_disp'].mean()

# # Define the monomial function to fit
# def monomial(X, fac, alpha):
#     return fac * X**alpha

# Fit the monomial function to the data
# popt, pcov = curve_fit(monomial, average_gamma_value.index, average_gamma_value.values, p0=[1, 1])

# # Extract the optimal parameters
# fac_opt, alpha_opt = popt
# print(f"Optimal parameters: fac = {fac_opt:.4f}, alpha = {alpha_opt:.4f}")
# # Plot the result
# fig, ax = plt.subplots(1, 3, figsize=(18, 6))
# fontsize=24
# ax[0].errorbar(average_max_advantage.index, average_max_advantage.values, yerr=variance_max_advantage.values, fmt='o', linestyle='-', color='b', ecolor='b', capsize=5)

# ax[1].plot(average_mean_disp.index, average_mean_disp.values, linestyle='None', color='y', marker='o', label='Mean Displacement')
# ax[1].plot(average_mean_disp.index, average_mean_disp.values / average_max_sqz.values, marker='o', color='r', linestyle='None', label='Ratio Disp/Sqz')
# ax[1].errorbar(average_max_sqz.index, average_max_sqz.values, yerr=3 * variance_max_sqz.values, fmt='o', linestyle='-', color='g', ecolor='g', capsize=5, label='Mean Squeezing')

# # Plot the fitted monomial function
# X_fit = np.linspace(min(average_gamma_value.index), max(average_gamma_value.index), 100)
# Y_fit = monomial(X_fit, fac_opt, alpha_opt)
# ax[2].errorbar(average_gamma_value.index, average_gamma_value.values, yerr=variance_gamma_value.values, fmt='o', linestyle='-', color='r', ecolor='r', capsize=5)
# ax[2].plot(X_fit, Y_fit, linestyle='--', color='b', label=f'Fit: {fac_opt:.4f} * X**{alpha_opt:.4f}')

# ax[0].legend(['Max Advantage', 'Gamma Value'], fontsize=fontsize*0.5)
# ax[1].legend(fontsize=fontsize*0.5)
# ax[2].legend(['Gamma Value', f'Factor: {fac_opt:.4f}, Power{alpha_opt:.4f}'], fontsize=fontsize*0.5)
# ax[0].set_xlabel('M', fontsize=fontsize*0.75)
# ax[1].set_xlabel('M', fontsize=fontsize*0.75)
# ax[2].set_xlabel('M', fontsize=fontsize*0.75)
# ax[0].set_ylabel('Advantage', fontsize=fontsize*0.75)
# ax[1].set_ylabel('Values', fontsize=fontsize*0.75)
# ax[2].set_ylabel('Average optimal gamma', fontsize=fontsize*0.75)
# ax[0].grid(True)
# ax[1].grid(True)
# ax[2].grid(True)
# ax[0].set_xticks(M)
# ax[1].set_xticks(M)
# ax[2].set_xticks(M)

# # Increase the size of the numbers on the x-axis and y-axis
# ax[0].tick_params(axis='both', which='major', labelsize=fontsize*0.5)
# ax[1].tick_params(axis='both', which='major', labelsize=fontsize*0.5)
# ax[2].tick_params(axis='both', which='major', labelsize=fontsize*0.5)

# # Set a global title
# fig.suptitle(f"Results for c={c_truncation_factor:.2f}*c_max, max_clique_size={clique_size} and {number_of_cliques} cliques", fontsize=fontsize)

# plt.savefig(os.path.join(plot_dir, f"average_max_advantage_{formatted_time}.svg"))
# plt.show()