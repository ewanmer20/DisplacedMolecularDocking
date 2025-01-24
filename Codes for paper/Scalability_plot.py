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

# Load the CSV file
csv_filename = 'results_20250120_202347.csv'  # Replace with the actual filename
csv_filepath = os.path.join(data_dir, csv_filename)
results_df = pd.read_csv(csv_filepath)

# Ensure the Mean_sqz column is numeric
results_df['Mean_sqz'] = results_df['Mean_sqz']

# Calculate the average Max_Advantage grouped by M
average_max_advantage = results_df.groupby('M')['Max_Advantage'].mean()
variance_max_advantage = results_df.groupby('M')['Max_Advantage'].var()
average_max_sqz = results_df.groupby('M')['Mean_sqz'].mean()
print(average_max_sqz)
variance_max_sqz = results_df.groupby('M')['Mean_sqz'].var()
average_gamma_value = results_df.groupby('M')['Gamma_value'].mean()
print(average_gamma_value)
variance_gamma_value = results_df.groupby('M')['Gamma_value'].var()

# Plot the result with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(average_max_advantage.index, average_max_advantage.values, yerr=variance_max_advantage.values, fmt='o', linestyle='-', color='b', ecolor='r', capsize=5, label='Max Advantage')
plt.errorbar(average_max_sqz.index, average_max_sqz.values, yerr=variance_max_sqz.values, fmt='o', linestyle='-', color='g', ecolor='r', capsize=5, label='Mean Squeezing')
plt.errorbar(average_gamma_value.index, average_gamma_value.values, yerr=variance_gamma_value.values, fmt='o', linestyle='-', color='r', ecolor='r', capsize=5, label='Gamma Value')
plt.xlabel('M')
plt.ylabel('Values')
plt.title('Average Max Advantage, Mean Squeezing, and Gamma Value as a function of M')
plt.legend()
plt.grid(True)

plt.show()
