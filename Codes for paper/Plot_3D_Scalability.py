import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

current_dir = os.path.dirname(__file__)
plot_dir = os.path.join(current_dir, 'Plots')
os.makedirs(plot_dir, exist_ok=True)
data_dir = os.path.join(current_dir, 'Data')
os.makedirs(data_dir, exist_ok=True)
os.chdir(plot_dir)

# Load the CSV file
csv_filename = 'results_20250202_124203.csv'  # Replace with the actual filename
csv_filepath = os.path.join(data_dir, csv_filename)
results_df = pd.read_csv(csv_filepath)


average_max_advantage = results_df.groupby(['M', 'Clique_size'])['Max_Advantage'].mean().reset_index()
variance_max_advantage = results_df.groupby(['M', 'Clique_size'])['Max_Advantage'].var().reset_index()
average_max_sqz = results_df.groupby(['M', 'Clique_size'])['Mean_sqz'].mean().reset_index()
variance_max_sqz = results_df.groupby(['M', 'Clique_size'])['Mean_sqz'].var().reset_index()
average_gamma_value = results_df.groupby(['M', 'Clique_size'])['Gamma_value'].mean().reset_index()
variance_gamma_value = results_df.groupby(['M', 'Clique_size'])['Gamma_value'].var().reset_index()
average_mean_disp = results_df.groupby(['M', 'Clique_size'])['Mean_disp'].mean().reset_index()
variance_mean_disp = results_df.groupby(['M', 'Clique_size'])['Mean_disp'].var().reset_index()
print(average_max_advantage)
# # Create a 3D bar plot for average_max_advantage with color mapping
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# colors = plt.cm.viridis(average_max_advantage['Max_Advantage'] / max(average_max_advantage['Max_Advantage']))
# ax.bar3d(average_max_advantage['M'], average_max_advantage['Clique_size'], np.zeros(len(average_max_advantage)), 1, 1, average_max_advantage['Max_Advantage'], color=colors)
# ax.set_xlabel('M')
# ax.set_ylabel('Clique_size')
# ax.set_zlabel('Average Max Advantage')
# ax.set_title('3D Bar Plot of Average Max Advantage')
# plt.savefig(os.path.join(plot_dir, 'average_max_advantage.png'))



# Create a 3D bar plot for average_max_advantage with color mapping and error bars
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.viridis(average_max_advantage['Max_Advantage'] / max(average_max_advantage['Max_Advantage']))
# colors = plt.cm.viridis(np.log(average_max_advantage['Max_Advantage'] / max(average_max_advantage['Max_Advantage'])))
ax.bar3d(average_max_advantage['M'], average_max_advantage['Clique_size'], np.zeros(len(average_max_advantage)), 1, 1, np.log(average_max_advantage['Max_Advantage']), color=colors)
for i in range(len(average_max_advantage)):
    ax.plot([average_max_advantage['M'][i], average_max_advantage['M'][i]], 
            [average_max_advantage['Clique_size'][i], average_max_advantage['Clique_size'][i]], 
            color='k')
ax.set_xlabel('M')
ax.set_ylabel('Clique_size')
ax.set_zlabel('Average Log(Max Advantage)')
ax.set_title('3D Bar Plot of Average Max Advantage')
plt.savefig(os.path.join(plot_dir, 'average_max_advantage.png'))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# colors = plt.cm.viridis(average_max_advantage['Max_Advantage'] / max(average_max_advantage['Max_Advantage']))
# ax.bar3d(average_max_advantage['M'], average_max_advantage['Clique_size'], np.zeros(len(average_max_advantage)), 1, 1, average_max_advantage['Max_Advantage'], color=colors)
# for i in range(len(average_max_advantage)):
#     ax.plot([average_max_advantage['M'][i], average_max_advantage['M'][i]], 
#             [average_max_advantage['Clique_size'][i], average_max_advantage['Clique_size'][i]], 
#             [average_max_advantage['Max_Advantage'][i] - variance_max_advantage['Max_Advantage'][i], 
#              average_max_advantage['Max_Advantage'][i] + variance_max_advantage['Max_Advantage'][i]], 
#             color='k')
# ax.set_xlabel('M')
# ax.set_ylabel('Clique_size')
# ax.set_zlabel('Average Max Advantage')
# ax.set_title('3D Bar Plot of Average Max Advantage')
# plt.savefig(os.path.join(plot_dir, 'average_max_advantage.png'))


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