#%%
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
#%%
# Initialize an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['M','Clique_size','Clique_size/M', 'Id', 'Max_Advantage', 'Gamma_value','Mean_disp', 'Mean_sqz','Adjacency_Matrix'])

c_truncation_factor=0.6 # Truncation factor for the photon number
erdos_renyi_prob = 0.4 # Probability of an edge in the Erdos-Renyi graph
number_of_cliques=40 # Number of cliques to generate per M 
gamma_truncation_factor=1 # Truncation factor for the gamma value
gamma_number=20 # Number of gamma values to generate 
r=[1.1,1.2,1.5,2,3]
clique_size_array=np.arange(6,24, 2).astype(int)
start_time_init = time.time() 
for i in range(len(r)):
    for j in range(len(clique_size_array)):
        for k in range(number_of_cliques):
            M=int(r[i]*clique_size_array[j])
            start_time = time.time()  # Start time of the iteration
            adj_matrix, clique_vector = generate_adjacency_matrix_with_clique(M, clique_size_array[j], erdos_renyi_prob)
            subgraph_1=np.array(clique_vector)
            Adj=adj_matrix
            c_max=find_max_c(Adj)
            # gamma_truncation_factor=0.2*(M)**(-2/8)
            gamma_truncation_factor=0.9
            gamma_array=np.linspace(0,gamma_truncation_factor,gamma_number)
            MaxCliqueProb_array=np.zeros(len(gamma_array))
            for l in range(len(gamma_array)):
                MaxCliqueProb_array[l]=probability_DGBS_subgraph(c_max*c_truncation_factor,gamma_array[l],Adj,subgraph_1)
            max_advantage,gamma_max_index=np.max(MaxCliqueProb_array), np.argmax(MaxCliqueProb_array)
            print(max_advantage / MaxCliqueProb_array[0])
            iteration_df = pd.DataFrame([{
                'M': M,
                'Clique_size':clique_size_array[j],
                'Clique_size/M':1/r[i],
                'Id': k,
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
            print(f"Iteration (i={i}, j={j},k={k}) took {duration:.2f} seconds")
end_time = time.time()  # End time of the iterations
print(f"Total time taken: {end_time - start_time_init:.2f} seconds")
now = datetime.now()
formatted_time = now.strftime("%Y%m%d_%H%M%S")
# Save the DataFrame to a CSV file in the Data subdirectory
filename = f"results_{formatted_time}.csv"
results_df.to_csv(os.path.join(data_dir, filename), index=False)

#%%
# formatted_time = "20251204_145100"
# formatted_time = "20251205_171328"
filename=f"results_{formatted_time}.csv"

results_df = pd.read_csv(os.path.join(data_dir, filename))

# Calculate the average Max_Advantage grouped by M and M/Clique_size
average_max_advantage = results_df.groupby(['Clique_size/M','Clique_size'])['Max_Advantage'].mean().unstack()
variance_max_advantage = results_df.groupby(['Clique_size/M', 'Clique_size'])['Max_Advantage'].var().unstack()
average_gamma_value = results_df.groupby(['Clique_size/M', 'Clique_size'])['Gamma_value'].mean().unstack()
variance_gamma_value = results_df.groupby(['Clique_size/M', 'Clique_size'])['Gamma_value'].var().unstack()
average_ratio_sqz_disp_mean = results_df.groupby(['Clique_size/M', 'Clique_size'])['Mean_sqz'].mean().unstack() / results_df.groupby(['Clique_size/M', 'Clique_size'])['Mean_disp'].mean().unstack()
variance_ratio_sqz_disp_mean = results_df.groupby(['Clique_size/M', 'Clique_size'])['Mean_sqz'].var().unstack() / results_df.groupby(['Clique_size/M', 'Clique_size'])['Mean_disp'].var().unstack()
# also compute average sqz and disp separately and their total
average_sqz = results_df.groupby(['Clique_size/M', 'Clique_size'])['Mean_sqz'].mean().unstack()
average_disp = results_df.groupby(['Clique_size/M', 'Clique_size'])['Mean_disp'].mean().unstack()
total_average_photons = average_sqz + average_disp

max_max_advantage = results_df.groupby(['Clique_size/M','Clique_size'])['Max_Advantage'].max().unstack()

# Plot the result (2x2 grid)
fig, ax = plt.subplots(2, 2, figsize=(14, 12))
fontsize = 24
# Legend font size for subplot figure (smaller than axis fontsize)
legend_fs = max(6, int(fontsize * 0.35))

# Plot average_max_advantage and fit each plotted series on ax[0] with a monomial model
def monomial(x, fac, alpha):
    return fac * x ** alpha

for ratio in max_max_advantage.index:
    x = max_max_advantage.columns.values.astype(float)
    y = max_max_advantage.loc[ratio].values.astype(float)
    # plot the raw series (top-left)
    ln = ax[0,0].plot(x, y, 'o-')
    color = ln[0].get_color()

    # Fit exponential model y = A * B**(C * x) to each series individually when possible
    def exp_model(x, A, B, C):
        return A * (B ** (C * x))

    if len(x) >= 2 and np.all(np.isfinite(y)):
        try:
            # Initial guess: if y>0, fit ln(y) = ln(A) + C * x * ln(B)
            if np.all(y > 0):
                # choose B_init = e to reduce to y = A * exp(C * x)
                ln_y = np.log(y)
                C_init, lnA_init = np.polyfit(x, ln_y, 1)
                A_init = np.exp(lnA_init)
                B_init = np.e
                # here C_init corresponds to C * ln(B_init) with ln(B_init)=1, so it's fine
            else:
                A_init = np.nanmean(y)
                B_init = np.e
                C_init = 0.0

            popt, _ = curve_fit(exp_model, x, y, p0=[A_init, B_init, C_init], maxfev=10000)
            A_fit, B_fit, C_fit = popt

            x_fit = np.linspace(x.min(), x.max(), 300)
            y_fit = exp_model(x_fit, A_fit, B_fit, C_fit)

            # Compute R^2 on original y values
            y_model = exp_model(x, A_fit, B_fit, C_fit)
            ss_res = np.sum((y - y_model) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

            # show fitted parameters in legend (display C and B)
            ax[0,0].plot(x_fit, y_fit, linestyle='--', color=color, linewidth=2,
                         label=f'C={C_fit:.3f}, B={B_fit:.3g}')
        except Exception:
            # fitting failed; skip
            pass

ax[0,0].legend(fontsize=legend_fs)
ax[0,0].set_xlabel(r'$|C|$', fontsize=fontsize * 0.75)
ax[0,0].set_ylabel(r'Average Max Advantage', fontsize=fontsize * 0.75)
ax[0,0].grid(True)
ax[0,0].set_yscale('log')
ax[0,0].set_xticks(max_max_advantage.columns)



# # Plot average_max_advantage
# for ratio in average_max_advantage.index:
#     ax[0].plot(average_max_advantage.columns, average_max_advantage.loc[ratio],'o-', label=f'R = {ratio:.2f}')

# ax[0].legend(fontsize=fontsize * 0.5)
# ax[0].set_xlabel(r'$|C|$', fontsize=fontsize * 0.75)
# ax[0].set_ylabel(r'Average Max Advantage', fontsize=fontsize * 0.75)
# ax[0].grid(True)
# ax[0].set_yscale('log')
# ax[0].set_xticks(average_max_advantage.columns)

# Plot average_gamma_value and fit model A / M**B for each ratio
def power_model(x, A, B):
    return A * x ** (-B)

for ratio in average_gamma_value.index:
    x = average_gamma_value.columns.values.astype(float)
    y = average_gamma_value.loc[ratio].values.astype(float)
    yerr = None
    try:
        yerr = variance_gamma_value.loc[ratio].values
    except Exception:
        yerr = None

    ax[0,1].errorbar(x, y, yerr=yerr, fmt='o-', label=f'R = {ratio:.2f}')

    # Fit A / M**B if data are suitable (positive y and enough points)
    if len(x) >= 2 and np.all(np.isfinite(y)):
        try:
            # Use log transform to get initial guess when possible
            if np.all(y > 0) and np.all(x > 0):
                log_coeffs = np.polyfit(np.log(x), np.log(y), 1)
                B_init = -log_coeffs[0]
                A_init = np.exp(log_coeffs[1])
            else:
                A_init = np.nanmean(y) * (x.mean() ** 1)
                B_init = 1.0

            popt, _ = curve_fit(power_model, x, y, p0=[A_init, B_init], maxfev=10000)
            A_fit, B_fit = popt
            x_fit = np.linspace(x.min(), x.max(), 300)
            y_fit = power_model(x_fit, A_fit, B_fit)

            # Compute R^2 on original points
            y_model = power_model(x, A_fit, B_fit)
            ss_res = np.sum((y - y_model) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

            # ax[0,1].plot(x_fit, y_fit, linestyle='--', color=ax[0,1].lines[-1].get_color(), label=f'Fit R={ratio:.2f}: A={A_fit:.3g}, B={B_fit:.3g}, R²={r2:.3f}')
            # ax[0,1].plot(x_fit, y_fit, linestyle='--', color=ax[0,1].lines[-1].get_color(), label=f'Fit:  Power={-B_fit:.3g}')
        except Exception:
            # fitting failed for this ratio; skip
            pass

ax[0,1].legend(fontsize=legend_fs)
ax[0,1].set_xlabel(r'$|C|$', fontsize=fontsize * 0.75)
ax[0,1].set_ylabel(r'$\gamma_{avg}$', fontsize=fontsize * 0.75)
ax[0,1].grid(True)
ax[0,1].set_xticks(average_gamma_value.columns)

for ratio in average_ratio_sqz_disp_mean.index:
    x = average_ratio_sqz_disp_mean.columns.values.astype(float)
    y = average_ratio_sqz_disp_mean.loc[ratio].values.astype(float)
    ax[1,0].plot(x, y, 'o-', label=f'R = {ratio:.2F}')

    # Fit A / M**B for n_sqz/n_disp if data valid
    if len(x) >= 2 and np.all(np.isfinite(y)):
        try:
            if np.all(y > 0) and np.all(x > 0):
                log_coeffs = np.polyfit(np.log(x), np.log(y), 1)
                B_init = -log_coeffs[0]
                A_init = np.exp(log_coeffs[1])
            else:
                A_init = np.nanmean(y) * (x.mean() ** 1)
                B_init = 1.0

            popt, _ = curve_fit(power_model, x, y, p0=[A_init, B_init], maxfev=10000)
            A_fit, B_fit = popt
            x_fit = np.linspace(x.min(), x.max(), 300)
            y_fit = power_model(x_fit, A_fit, B_fit)

            y_model = power_model(x, A_fit, B_fit)
            ss_res = np.sum((y - y_model) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

            # ax[1,0].plot(x_fit, y_fit, linestyle='--', color=ax[1,0].lines[-1].get_color(), label=f'Fit R={ratio:.2f}: A={A_fit:.3g}, B={B_fit:.3g}, R²={r2:.3f}')
            ax[1,0].plot(x_fit, y_fit, linestyle='--', color=ax[1,0].lines[-1].get_color(), label=f'Fit: Power={-B_fit:.3g}')
        except Exception:
            pass
# Add a dotted horizontal reference line at y=1
# ax[1,0].axhline(1, color='k', linestyle=':', linewidth=1.5)

ax[1,0].legend(fontsize=legend_fs)
ax[1,0].set_xlabel(r'$|C|$', fontsize=fontsize * 0.75)
ax[1,0].set_ylabel(r'$\bar{n}_{sqz}/\bar{n}_{disp}$', fontsize=fontsize * 0.75)
ax[1,0].grid(True)
ax[1,0].set_xticks(average_ratio_sqz_disp_mean.columns)

# Increase the size of the numbers on the x-axis and y-axis
ax[0,0].tick_params(axis='both', which='major', labelsize=fontsize * 0.5)
ax[0,1].tick_params(axis='both', which='major', labelsize=fontsize * 0.5)
ax[1,0].tick_params(axis='both', which='major', labelsize=fontsize * 0.5)
ax[1,1].tick_params(axis='both', which='major', labelsize=fontsize * 0.5)

# Fourth subplot (bottom-right): total average photon number (n_sqz + n_disp) and monomial fit vs |C|
for ratio in total_average_photons.index:
    x = total_average_photons.columns.values.astype(float)
    y = total_average_photons.loc[ratio].values.astype(float)
    ln = ax[1,1].plot(x, y, 'o-', label=f'R = {ratio:.2f}')
    color = ln[0].get_color()

    if len(x) >= 2 and np.all(np.isfinite(y)):
        try:
            if np.all(y > 0) and np.all(x > 0):
                log_coeffs = np.polyfit(np.log(x), np.log(y), 1)
                alpha_init = log_coeffs[0]
                fac_init = np.exp(log_coeffs[1])
            else:
                fac_init = np.nanmean(y)
                alpha_init = 0.0

            popt, _ = curve_fit(monomial, x, y, p0=[fac_init, alpha_init], maxfev=10000)
            fac_fit, alpha_fit = popt
            x_fit = np.linspace(x.min(), x.max(), 300)
            y_fit = monomial(x_fit, fac_fit, alpha_fit)

            y_model = monomial(x, fac_fit, alpha_fit)
            ss_res = np.sum((y - y_model) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

            # ax[1,1].plot(x_fit, y_fit, linestyle='--', color=color, linewidth=2, label=rf'Fit R={ratio:.2f}, alpha={alpha_fit:.3f}, R²={r2:.3f}')
            ax[1,1].plot(x_fit, y_fit, linestyle='--', color=color, linewidth=2, label=rf' alpha={alpha_fit:.3f}')
        except Exception:
            pass

ax[1,1].legend(fontsize=legend_fs)
ax[1,1].set_xlabel(r'$|C|$', fontsize=fontsize * 0.75)
ax[1,1].set_ylabel(r'$\bar{n}_{tot}$', fontsize=fontsize * 0.75)
ax[1,1].grid(True)
ax[1,1].set_xticks(total_average_photons.columns)


# Set a global title
fig.suptitle(f"Average Max Advantage and Gamma Value for Different M/Clique_size Ratios", fontsize=fontsize)

plt.savefig(os.path.join(plot_dir, f"average_max_advantage_and_gamma_value_ratios_{formatted_time}.svg"))
plt.show()
# %%
