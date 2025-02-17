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
from scipy.optimize import curve_fit

current_dir = os.path.dirname(__file__)
plot_dir = os.path.join(current_dir, 'Plots')
os.makedirs(plot_dir, exist_ok=True)
data_dir = os.path.join(current_dir, 'Data')
os.makedirs(data_dir, exist_ok=True)
os.chdir(plot_dir)

# Initialize an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['Clique_size','M', 'Id', 'Max_Advantage', 'Gamma_value','Mean_disp', 'Mean_sqz','Adjacency_Matrix'])

clique_size = 20 # Size of the cliques
offset_M=2 # Offset for the M values
c_truncation_factor=0.5 # Truncation factor for the photon number
erdos_renyi_prob = 0.2 # Probability of an edge in the Erdos-Renyi graph
number_of_M=14 # Total number of M values to generate
number_of_cliques=25 # Number of cliques to generate per M 
gamma_truncation_factor=1 # Truncation factor for the gamma value
gamma_number=200 # Number of gamma values to generate 
clique_size_array=np.arange(8,36, 2).astype(int)
for clique_size in clique_size_array:
    M=np.arange(clique_size+offset_M, clique_size+number_of_M+offset_M-1, 2).astype(int)

    start_time_init = time.time() 
    for i in range(len(M)):
        for j in range(number_of_cliques):
            start_time = time.time()  # Start time of the iteration
            adj_matrix, clique_vector = generate_adjacency_matrix_with_clique(M[i], clique_size, erdos_renyi_prob)
            subgraph_1=np.array(clique_vector)
            Adj=adj_matrix
            c_max=find_max_c(Adj)
            gamma_array=np.linspace(0,gamma_truncation_factor,gamma_number)
            MaxCliqueProb_array=np.zeros(len(gamma_array))
            for k in range(len(gamma_array)):
                MaxCliqueProb_array[k]=probability_DGBS_subgraph(c_max*c_truncation_factor,gamma_array[k],Adj,subgraph_1)
            gamma_max_index=np.argmax(MaxCliqueProb_array)
            max_advantage=np.max(MaxCliqueProb_array)
            # print(max_advantage / MaxCliqueProb_array[0])
            iteration_df = pd.DataFrame([{
                'Clique_size':clique_size,
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
formatted_time = now.strftime("%Y%m%d_%H%M%S")
# Save the DataFrame to a CSV file in the Data subdirectory
filename = f"results_{formatted_time}.csv"
results_df.to_csv(os.path.join(data_dir, filename), index=False)

# Calculate the average Max_Advantage grouped by M and Clique_size
average_max_advantage = results_df.groupby(['M', 'Clique_size'])['Max_Advantage'].mean().reset_index()
variance_max_advantage = results_df.groupby(['M', 'Clique_size'])['Max_Advantage'].var().reset_index()
average_max_sqz = results_df.groupby(['M', 'Clique_size'])['Mean_sqz'].mean().reset_index()
variance_max_sqz = results_df.groupby(['M', 'Clique_size'])['Mean_sqz'].var().reset_index()
average_gamma_value = results_df.groupby(['M', 'Clique_size'])['Gamma_value'].mean().reset_index()
variance_gamma_value = results_df.groupby(['M', 'Clique_size'])['Gamma_value'].var().reset_index()
average_mean_disp = results_df.groupby(['M', 'Clique_size'])['Mean_disp'].mean().reset_index()
variance_mean_disp = results_df.groupby(['M', 'Clique_size'])['Mean_disp'].var().reset_index()
print(average_max_advantage)
# Create a 3D bar plot for average_max_advantage with color mapping
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.viridis(average_max_advantage['Max_Advantage'] / max(average_max_advantage['Max_Advantage']))
ax.bar3d(average_max_advantage['M'], average_max_advantage['Clique_size'], np.zeros(len(average_max_advantage)), 1, 1, average_max_advantage['Max_Advantage'], color=colors)
ax.set_xlabel('M')
ax.set_ylabel('Clique_size')
ax.set_zlabel('Average Max Advantage')
ax.set_title('3D Bar Plot of Average Max Advantage')
plt.savefig(os.path.join(plot_dir, 'average_max_advantage.png'))


# ...existing code...

# Create a 3D bar plot for average_max_advantage with color mapping and error bars
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.viridis(average_max_advantage['Max_Advantage'] / max(average_max_advantage['Max_Advantage']))
ax.bar3d(average_max_advantage['M'], average_max_advantage['Clique_size'], np.zeros(len(average_max_advantage)), 1, 1, average_max_advantage['Max_Advantage'], color=colors)
for i in range(len(average_max_advantage)):
    ax.plot([average_max_advantage['M'][i], average_max_advantage['M'][i]], 
            [average_max_advantage['Clique_size'][i], average_max_advantage['Clique_size'][i]], 
            [average_max_advantage['Max_Advantage'][i] - variance_max_advantage['Max_Advantage'][i], 
             average_max_advantage['Max_Advantage'][i] + variance_max_advantage['Max_Advantage'][i]], 
            color='k')
ax.set_xlabel('M')
ax.set_ylabel('Clique_size')
ax.set_zlabel('Average Max Advantage')
ax.set_title('3D Bar Plot of Average Max Advantage')
plt.savefig(os.path.join(plot_dir, 'average_max_advantage.png'))


# Create a 3D bar plot for average_max_sqz with color mapping and error bars
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.viridis(average_max_sqz['Mean_sqz'] / max(average_max_sqz['Mean_sqz']))
ax.bar3d(average_max_sqz['M'], average_max_sqz['Clique_size'], np.zeros(len(average_max_sqz)), 1, 1, average_max_sqz['Mean_sqz'], color=colors)
for i in range(len(average_max_sqz)):
    ax.plot([average_max_sqz['M'][i], average_max_sqz['M'][i]], 
            [average_max_sqz['Clique_size'][i], average_max_sqz['Clique_size'][i]], 
            [average_max_sqz['Mean_sqz'][i] - variance_max_sqz['Mean_sqz'][i], 
             average_max_sqz['Mean_sqz'][i] + variance_max_sqz['Mean_sqz'][i]], 
            color='k')
ax.set_xlabel('M')
ax.set_ylabel('Clique_size')
ax.set_zlabel('Average Max Sqz')
ax.set_title('3D Bar Plot of Average Max Sqz')
plt.savefig(os.path.join(plot_dir, 'average_max_sqz.png'))


# Create a 3D bar plot for average_gamma_value with color mapping and error bars
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.viridis(average_gamma_value['Gamma_value'] / max(average_gamma_value['Gamma_value']))
ax.bar3d(average_gamma_value['M'], average_gamma_value['Clique_size'], np.zeros(len(average_gamma_value)), 1, 1, average_gamma_value['Gamma_value'], color=colors)
for i in range(len(average_gamma_value)):
    ax.plot([average_gamma_value['M'][i], average_gamma_value['M'][i]], 
            [average_gamma_value['Clique_size'][i], average_gamma_value['Clique_size'][i]], 
            [average_gamma_value['Gamma_value'][i] - variance_gamma_value['Gamma_value'][i], 
             average_gamma_value['Gamma_value'][i] + variance_gamma_value['Gamma_value'][i]], 
            color='k')
ax.set_xlabel('M')
ax.set_ylabel('Clique_size')
ax.set_zlabel('Average Gamma Value')
ax.set_title('3D Bar Plot of Average Gamma Value')
plt.savefig(os.path.join(plot_dir, 'average_gamma_value.png'))


# Create a 3D bar plot for average_mean_disp with color mapping and error bars
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.viridis(average_mean_disp['Mean_disp'] / max(average_mean_disp['Mean_disp']))
ax.bar3d(average_mean_disp['M'], average_mean_disp['Clique_size'], np.zeros(len(average_mean_disp)), 1, 1, average_mean_disp['Mean_disp'], color=colors)
for i in range(len(average_mean_disp)):
    ax.plot([average_mean_disp['M'][i], average_mean_disp['M'][i]], 
            [average_mean_disp['Clique_size'][i], average_mean_disp['Clique_size'][i]], 
            [average_mean_disp['Mean_disp'][i] - variance_mean_disp['Mean_disp'][i], 
             average_mean_disp['Mean_disp'][i] + variance_mean_disp['Mean_disp'][i]], 
            color='k')
ax.set_xlabel('M')
ax.set_ylabel('Clique_size')
ax.set_zlabel('Average Mean Disp')
ax.set_title('3D Bar Plot of Average Mean Disp')
plt.savefig(os.path.join(plot_dir, 'average_mean_disp.png'))
plt.show()
