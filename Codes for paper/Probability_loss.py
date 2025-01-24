import sys
import time
# Add the Prakash folder to sys.path
# Add the Script_DGBS directory to sys.path
sys.path.append(r'C:\Users\em1120\DisplacedMolecularDocking')
from datetime import datetime
# Import calc_unitary from scripts
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Scripts_DGBS.probability_max_clique import *

current_dir = os.path.dirname(__file__)
plot_dir = os.path.join(current_dir, 'Plots')
os.makedirs(plot_dir, exist_ok=True)
data_dir = os.path.join(current_dir, 'Data')
os.makedirs(data_dir, exist_ok=True)
os.chdir(plot_dir)

Adj=np.array([
  [0, 1, 1, 1, 0, 0],
  [1, 0, 1, 1, 0, 1],
  [1, 1, 0, 1, 0, 0],
  [1, 1, 1, 0, 1, 0],
  [0, 0, 0, 1, 0, 1],
  [0, 1, 0, 0, 1, 0]
])
subgraph_1=np.array([1,1,1,1,0,0])
loss=np.linspace(0,1,50)
gamma_val=[0,0.2,0.5,1,2]
c=0.08
cutoff=3
results_df = pd.DataFrame(columns=['Loss','c', 'Gamma_value','Mean_photon_sqz','Mean_photon_disp','Probability','Adjacency_Matrix'])
loss_result=[]
start=time.time()
c_max=find_max_c(Adj)
c_trunc=0.5
for j in range(len(gamma_val)):
    for i in range(len(loss)):
        start_i=time.time()
        lossy_prob=probability_array_DGBS(c_max*c_trunc, gamma_val[j], Adj,loss=loss[i], cutoff=cutoff)[tuple(subgraph_1)]
        loss_result.append(lossy_prob)
        serie_df = pd.DataFrame([{
            'Loss': loss[i],
            'c': c_max*c_trunc,
            'Gamma_value': gamma_val[j],
            'Mean_photon_sqz':sqz_photon_number(c_max*c_trunc,Adj),
            'Mean_photon_disp':disp_photon_number(c_max*c_trunc,gamma_val[j],Adj),
            'Probability': lossy_prob,
            'Adjacency_Matrix': np.ravel(Adj)
        }])
        print(f"Loss: {loss[i]:.2f}, Probability: {lossy_prob:.2e}")
        results_df = pd.concat([results_df, serie_df], ignore_index=True)
        end_i=time.time()
        print("Time taken for compute lossy_prob:", end_i-start_i)
    end=time.time()
print("Time taken for compute lossy_prob:", end-start)

now = datetime.now()
formatted_time = now.strftime("%Y%m%d_%H%M%S")
# Save the DataFrame to a CSV file in the Data subdirectory
filename = f"results_{formatted_time}.csv"

# Plot the probability loss as a function of loss grouped by different values of gamma_value
plt.figure(figsize=(10, 6))
for gamma in results_df['Gamma_value'].unique():
    subset = results_df[results_df['Gamma_value'] == gamma]
    plt.plot(subset['Loss'], subset['Probability'], marker='o', linestyle='-', label=f'Gamma = {gamma:.2f}')

plt.xlabel('Loss')
plt.ylabel('Probability')
plt.title('Probability Loss as a function of Loss grouped by Gamma Value')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig(os.path.join(plot_dir, f"lossy_GBS_{formatted_time}.svg"))
plt.show()


    
