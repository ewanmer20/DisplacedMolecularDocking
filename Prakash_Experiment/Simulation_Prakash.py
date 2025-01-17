
import sys
# Add the Prakash folder to sys.path
# Add the Script_DGBS directory to sys.path
sys.path.append(r'C:\Users\em1120\DisplacedMolecularDocking')
sys.path.append(r'C:\Users\em1120\DisplacedMolecularDocking\Prakash')

# Import calc_unitary from scripts
from Prakash.scripts.calc_unitary import calc_mesh
import os
import copy
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt
import networkx as nx
from Scripts_DGBS.utils import mean_nsqz
from Scripts_DGBS.DGBS_ArbitraryGraph_class import *
from Scripts_DGBS.probability_max_clique import generate_adjacency_matrix_with_clique, general_reduction,probability_max_clique_DGBS
from colorsys import hls_to_rgb

current_dir = os.path.dirname(__file__)
plot_dir = os.path.join(current_dir, 'Plots')
os.makedirs(plot_dir, exist_ok=True)
os.chdir(plot_dir)

def colorize(fz):

    """
    The original colorize function can be found at:
    https://stackoverflow.com/questions/17044052/mathplotlib-imshow-complex-2d-array
    by the user nadapez.
    """
    
    r = np.log2(1. + np.abs(fz))
    
    h = np.angle(fz)/(2*np.pi)
    l = 1 - 0.45**(np.log(1+r)) 
    s = 1

    c = np.vectorize(hls_to_rgb)(h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (m,n,3)
    c = np.rot90(c.transpose(2,1,0), 1) # Change shape to (m,n,3) and rotate 90 degrees
    
    return c

def tune_c_custom(target_nsqz,Adj,weights):
    """

    :param alpha: the alpha at the input of the adjacency matrix
    :param target_nsqz:  positive number representing  target mean photon n for the squeezing
    :param Adjtot:  adjacency matrix of the total graph
    :param nsubpsace:  dimension of the considered subspace
    :return:
    """
    def cost(c,target_nsqz,Adj):
        BIG=c*np.diag(weights)@ Adj @ np.diag(weights)
        return np.abs(target_nsqz-mean_nsqz(BIG))
    res=minimize_scalar(cost,args=(target_nsqz,Adj))
    return res.x

def unitary_optimization(Adj,weights,target_nsqz,max_current=20):
    """Tune the mesh of current of the chip and the squeezing parameter to reach the target weighed graphs
    Adj: Adjacency matrix of the graph (symmetrix array of 0 and 1, with the diagonal elements being null)
    weights: list of weights used to build the BIG matrix with the same length as the sample
    target_nsqz:  positive number representing  target mean photon n for the squeezing
    max_current: Maximum current of the chip in mA
    """
    # def cost_function(params,Adj,ideal_weighted_graph):
    #     """Cost function for the DGBS
    # params: list of parameters including the current settings and the squeezing parameters (length of the list is 3*Adj.shape[0])
    # Adj: Adjacency matrix of the graph (symmetrix array of 0 and 1, with the diagonal elements being null)
    # ideal_weighted_graph: the target weighted graph 
    # """
    #     tanhr_list=np.sort(np.array(params[0:Adj.shape[0]]))
    #     current_set=np.array(params[Adj.shape[0]:]).reshape(Adj.shape[0],Adj.shape[0])
    #     U = calc_mesh(current_set)
    #     real_matrix=U@np.diag(tanhr_list)@U.T
    #     return np.linalg.norm(real_matrix-ideal_weighted_graph,'fro')/Adj.shape[0]**2

    def cost_function(params,Adj,ideal_weighted_graph):
        """Cost function for the DGBS
    params: list of parameters including the current settings and the squeezing parameters (length of the list is 3*Adj.shape[0])
    Adj: Adjacency matrix of the graph (symmetrix array of 0 and 1, with the diagonal elements being null)
    ideal_weighted_graph: the target weighted graph 
    """
        tanhr_list=np.sort(np.array(params[0:Adj.shape[0]]))
        current_set=np.array(params[Adj.shape[0]:]).reshape(Adj.shape[0],Adj.shape[0])
        U = calc_mesh(current_set)
        real_matrix=U@np.diag(tanhr_list)@U.T
        return np.linalg.norm(np.angle(real_matrix),'fro')/Adj.shape[0]**2

    c=tune_c_custom(target_nsqz,Adj,weights)
    ideal_weighted_graph=c*np.diag(weights)@ Adj @ np.diag(weights)
    current_mesh_init=np.random.rand(Adj.shape[0],Adj.shape[0])*max_current
    tanh_array_init=np.random.rand(Adj.shape[0])
    x0=np.concatenate([tanh_array_init,np.ravel(current_mesh_init)])
       # Define bounds for the parameters
    bounds = [(0.5, 1)] * Adj.shape[0] + [(0, max_current)] * (Adj.shape[0] * Adj.shape[0])
    res = minimize(cost_function, x0, args=(Adj, ideal_weighted_graph), method='L-BFGS-B', bounds=bounds, options={'disp': True,'maxiter':10})
    return res,c


total_size = 10
clique_size = 4
erdos_renyi_prob = 0.3
adj_matrix, clique_vector = generate_adjacency_matrix_with_clique(total_size, clique_size, erdos_renyi_prob)
print("Adjacency Matrix:\n", adj_matrix)
print("Clique Vector:\n", clique_vector)

Adj=adj_matrix
subgraph_1=np.array(clique_vector)
Adj_clique=general_reduction(Adj,subgraph_1)
n_subspace=Adj.shape[0]
Id = np.eye(n_subspace)
c_array=np.linspace(0,0.3,200)
gamma_array=np.linspace(0,1,200)
MaxCliqueProb_array=np.zeros((len(gamma_array),len(c_array)))
Norm_array=np.zeros((len(gamma_array),len(c_array)))
Loop_hafnian_array=np.zeros((len(gamma_array),len(c_array)))
for i in range(len(gamma_array)):
    for j in range(len(c_array)):
        MaxCliqueProb_array[i,j]=probability_max_clique_DGBS(c_array[j],gamma_array[i],Adj,Adj_clique)

# Create a graph from the adjacency matrix
G = nx.from_numpy_array(Adj)

# Plot the graph
plt.figure(figsize=(8, 8))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
plt.title('Graph Representation')
plt.savefig(f'Graph{total_size}_clique_size{clique_size}_erdos_prob{erdos_renyi_prob}.svg',transparent=True)
plt.rcParams.update({'font.size': 24})
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(24, 24))
im=ax.imshow(MaxCliqueProb_array,origin='lower')

plt.colorbar(im,fraction=0.05)
ax.set_title('Probability Max Clique')
contour = ax.contour(MaxCliqueProb_array, levels=8, colors='white',origin='lower')
ax.set_xlabel('c') 
ax.set_ylabel('gamma') 
ax.clabel(contour, inline=True, fontsize=8, colors='white')
# Set the tick positions and labels for the original matrix
ax.set_xticks(np.linspace(0, len(c_array) - 1, 5))
ax.set_xticklabels(np.round(np.linspace(c_array[0], c_array[-1], 5), 2))
ax.set_yticks(np.linspace(0, len(gamma_array) - 1, 5))
ax.set_yticklabels(np.round(np.linspace(gamma_array[0], gamma_array[-1], 5), 2))
plt.savefig(f'Figure{total_size}_clique_size{clique_size}_erdos_prob{erdos_renyi_prob}.svg')

weights=np.ones(Adj.shape[0])
res,c=unitary_optimization(Adj,weights,2)

optimal_cost = res.fun
print("Optimal cost function value:", optimal_cost)
print("Optimal parameters:", res.x)
print("Optimal c:", c)
tanhr_list=np.sort(np.array(res.x[0:Adj.shape[0]]))
current_set=np.array(res.x[Adj.shape[0]:]).reshape(Adj.shape[0],Adj.shape[0])
U = calc_mesh(current_set)
real_matrix=U@np.diag(tanhr_list)@U.T
print("Real Matrix:\n", real_matrix)

magnitude = np.abs(real_matrix)
phase = np.angle(real_matrix)
magnitude_ideal = np.abs(Adj)
phase_ideal = np.angle(Adj)


fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(10,10))
im0=ax[0,0].imshow(magnitude,cmap='plasma')
ax[0,0].set_title("Magnitude Prakash",fontsize=28)
fig.colorbar(im0,ax=ax[0,0])
im1=ax[0,1].imshow(phase,cmap='coolwarm')
ax[0,1].set_title("Phase Prakash",fontsize=28)
fig.colorbar(im1,ax=ax[0,1])
im2=ax[1,0].imshow(magnitude_ideal,cmap='plasma')
ax[1,0].set_title("Magnitude Target",fontsize=28)
fig.colorbar(im2,ax=ax[1,0])
im3=ax[1,1].imshow(phase_ideal,cmap='coolwarm')
ax[1,1].set_title("Phase Target",fontsize=28)
fig.colorbar(im3,ax=ax[1,1])
plt.show()














































# Adj_clique=general_reduction(Adj,subgraph_1)
# n_subspace=Adj.shape[0]
# Id = np.eye(n_subspace)
# c_array=np.linspace(0,0.075,200)
# gamma_array=np.linspace(0,0.5,200)
# MaxCliqueProb_array=np.zeros((len(gamma_array),len(c_array)))
# Norm_array=np.zeros((len(gamma_array),len(c_array)))
# Loop_hafnian_array=np.zeros((len(gamma_array),len(c_array)))
# for i in range(len(gamma_array)):
#     for j in range(len(c_array)):
#         MaxCliqueProb_array[i,j]=probability_max_clique_DGBS(c_array[j],gamma_array[i],Adj,Adj_clique)

# #  Plot the adjacency matrix
# plt.imshow(Adj, cmap='Greys', interpolation='none')
# plt.title('Adjacency Matrix')
# plt.colorbar(label='Edge Presence')
# plt.xlabel('Node Index')
# plt.ylabel('Node Index')
# plt.savefig(f'Adjacency_Matrix_size{total_size}_clique_size{clique_size}_erdos_prob{erdos_renyi_prob}.svg')
# # Create a graph from the adjacency matrix
# G = nx.from_numpy_array(Adj)

# # Plot the graph
# plt.figure(figsize=(8, 8))
# nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
# plt.title('Graph Representation')
# plt.savefig(f'Graph{total_size}_clique_size{clique_size}_erdos_prob{erdos_renyi_prob}.svg',transparent=True)
# plt.rcParams.update({'font.size': 40})
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(24, 24))
# im=ax.imshow(MaxCliqueProb_array,origin='lower')

# plt.colorbar(im,fraction=0.05)
# ax.set_title('Probability Max Clique')
# contour = ax.contour(MaxCliqueProb_array, levels=8, colors='white',origin='lower')
# ax.set_xlabel('c') 
# ax.set_ylabel('gamma') 
# ax.clabel(contour, inline=True, fontsize=8, colors='white')
# # Set the tick positions and labels for the original matrix
# ax.set_xticks(np.linspace(0, len(c_array) - 1, 5))
# ax.set_xticklabels(np.round(np.linspace(c_array[0], c_array[-1], 5), 2))
# ax.set_yticks(np.linspace(0, len(gamma_array) - 1, 5))
# ax.set_yticklabels(np.round(np.linspace(gamma_array[0], gamma_array[-1], 5), 2))
# plt.savefig(f'Figure{total_size}_clique_size{clique_size}_erdos_prob{erdos_renyi_prob}.svg')
# plt.show()