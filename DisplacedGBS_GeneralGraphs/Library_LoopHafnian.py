import matplotlib.pyplot as plt
import numpy as np

from Library_Analysis import*
import thewalrus as thw
import random
import networkx as nx
from scipy.special import factorial
def random_subgraph(adj,nvertices):
    """
    Return a subgraph with nvertices nodes from a given graph with a uniform probability distribution
    :param adj: adjacency matrix of the graph
    :param nvertices: number of nodes of the subgraph
    :return:
    """
    indexes=np.array(random.sample(range(len(adj)),nvertices)).astype(np.int64)
    seq=np.zeros(len(adj),dtype=np.int64)
    for i in range(len(indexes)):
        seq[indexes[i]]=1
    return thw.reduction(adj,seq)

def random_subgraph_list(adj,nvertices):
    """
    Return a list of subgraphs with nvertices nodes from a given graph with a uniform probability distribution
    :param adj: adjacency matrix of the graph
    :param nvertices: number of nodes of the subgraphs
    :return:
    """
    indexes=np.array(random.sample(range(len(adj)),nvertices)).astype(np.int64)
    seq=np.zeros(len(adj),dtype=np.int64)
    for i in range(len(indexes)):
        seq[indexes[i]]=1
    return seq

def max_clique_graph(adj,weights=None):
    """
    Return the maximal clique of a graph
    :param adj: adjacency matrix of the graph
    :param weights: 1D array representing the weights of each nodes
    :return: maximal clique
    """
    if weights is None:
        weights=np.ones(len(adj))
    clique_max_seq, clique_weight = find_max_clique(adj, weights, networkx_conv=False)
    max_clique = thw.reduction(adj, clique_max_seq.astype(np.int64))
    return max_clique

def max_clique_list(adj,weights=None):
    """
    Return the list of max cliques
    :param adj: adjacency matrix of the graph
    :param weights: 1D array representing the weights of each nodes
    :return: list of maximal cliques
    """
    if weights is None:
        weights=np.ones(len(adj))
    clique_max_seq, clique_weight = find_max_clique(adj, weights, networkx_conv=False)
    return clique_max_seq.astype(np.int64)

def random_adj(nvertices,prob_edges):
    """
    Generate a random Erdos-Renyi graph
    :param nvertices: Number of vertices of the graph
    :param prob_edges: Edge probability
    :return: Adjacency matrix
    """
    graph=nx.erdos_renyi_graph(n=nvertices,p=prob_edges,directed=False)
    adj=nx.to_numpy_array(graph)
    return adj

def is_a_clique(adj):
    if np.sum(adj)==len(adj)**2:
        return True
    else:
        return False
def factorial_prod(v):
    """
    Return the product of factorials of the vector v
    :param v: numpy array v
    :return:
    """
    return np.prod(factorial(v))
def probability_c(adj_tot,subgraph,alpha,tanhrmax,loop=True):
    """
    Return the probability of obtaining a subgraph from a larger graph from a GBS experiment
    :param adj_tot: Adjacency matrix of the total graph
    :param subgraph: 1D Numpy array with one and zeros element selecting the subgraph from adj_tot
    :param loop: Tells if we are running a displaced GBS experiment or standard GBS experiment
    :return: the probability
    """
    Id = np.eye(len(adj_tot))
    c=tanhrmax/(max(np.abs(np.linalg.eigvalsh(adj_tot))))
    weights=make_potential_vect()[:len(adj_tot)]
    omega=c*(Id+alpha*np.diag(weights))
    BIG=omega@adj_tot@omega
    Sigma_Qinv = np.block([[Id, -BIG], [-BIG, Id]])
    Sigma_Q = inv(Sigma_Qinv)
    gamma =np.block([[omega, np.zeros((len(adj_tot),len(adj_tot)))], [np.zeros((len(adj_tot),len(adj_tot))), omega]])@ np.concatenate(((1+weights)**3,(1+weights)**3))
    d_alpha = (Sigma_Q @ gamma)
    if loop==True:
        norm=np.exp(-0.5*np.dot(d_alpha,Sigma_Qinv@d_alpha))/np.sqrt(np.linalg.det(Sigma_Q))
        reduced_adj=thw.reduction(BIG,subgraph)
        reduced_diag=thw.reduction(gamma,subgraph)
        np.fill_diagonal(reduced_adj, reduced_diag)
        return norm*thw.hafnian(reduced_adj,loop=True)**2,d_alpha

    if loop==False:
        norm=1/np.sqrt(np.linalg.det(Sigma_Q))
        reduced_adj=thw.reduction(BIG,subgraph)
        return norm*thw.hafnian(reduced_adj,loop=False)**2

def make_generalized_omega(c,alpha):
    """""
    function to generate a more generalized rescaling matrix omega, as defined in Banchi et. where c depends on the mode
    al.
    c is a numpy 1D array  of positive floats that controls the amount squeezing required
    alpha is the strength of the weight potentials in the matrix

    returns a 2-d numpy array
    """""
    big_potentials = make_potential_vect()[:len(c)]
    omega = np.diag(c) @ (np.eye(len(big_potentials)) +alpha * np.diag(big_potentials))
    return omega

def probability_BIG(adj_tot,subgraph,c,v,alpha,nsubspace,loop=True):
    """
    Return the probability of obtaining a subgraph from a larger graph from a GBS experiment
    :param adj_tot: Adjacency matrix of the total graph
    :param subgraph: 1D Numpy array with one and zeros element selecting the subgraph from adj_tot
    :param loop: Tells if we are running a displaced GBS experiment or standard GBS experiment
    :return: the probability
    """
    Id = np.eye(nsubspace)
    tanhr=return_r(adj_tot,c,v,alpha)
    nsqz=np.sum(np.divide((tanhr)**2,1-(tanhr)**2))
    omega = make_generalized_omega(c, alpha)[:nsubspace, :nsubspace]
    BIG = omega @ Adj @ omega + np.diag(v) @ np.eye(nsubspace)
    Sigma_Qinv = np.block([[Id, -BIG], [-BIG, Id]])
    Sigma_Q = inv(Sigma_Qinv)
    gamma = np.concatenate([np.diag(omega), np.diag(omega)])
    d_alpha = (Sigma_Q @ gamma)
    ncoh = np.sum(d_alpha ** 2)
    if loop==True:
        norm=np.exp(-0.5*np.dot(d_alpha,Sigma_Qinv@d_alpha))/np.sqrt(np.linalg.det(Sigma_Q))
        reduced_BIG=thw.reduction(BIG,subgraph)
        reduced_diag=thw.reduction(gamma,subgraph)
        np.fill_diagonal(reduced_BIG, reduced_diag)
        return norm*thw.hafnian(reduced_BIG,loop=True)**2,nsqz,ncoh

    if loop==False:
        norm=1/np.sqrt(np.linalg.det(Sigma_Q))
        reduced_BIG=thw.reduction(BIG,subgraph)
        return norm*thw.hafnian(reduced_BIG,loop=False)**2,nsqz



# # sq_min=0.2
# # sq_max=0.4
# # nsubspace=24
# # alpha=1.5
# # cwd = 'big\\adj_mat_tau1.1_.csv'
# # Adj= log_data(cwd)
# # target_ncoh=[0.001,0.050,0.100,0.500,1,1.5]
# # problhaf_hist=np.zeros(6)
# # probhaf_hist=np.zeros(6)
# # nsqz1_hist=np.zeros(6)
# # ncoh_hist=np.zeros(6)
# weigths= make_potential_vect()
# max_clique=max_clique_list(Adj,weigths)
# for i in range (len(target_ncoh)):
#     c=log_data('Parameters_c_v\\TaceAs\\'+'sqmin={:.1f}'.format(sq_min)+'sqmax={:.1f}'.format(sq_max)+'dim={:.1f}'.format(nsubspace)+'ncoh={:.3f}'.format(target_ncoh[i])+'alpha={:.2f}'.format(alpha)+'cparameters.csv').reshape((nsubspace,))
#     v=log_data('Parameters_c_v\\TaceAs\\' + 'sqmin={:.1f}'.format(sq_min) + 'sqmax={:.1f}'.format(sq_max) + 'dim={:.1f}'.format(nsubspace) + 'ncoh={:.3f}'.format(target_ncoh[i])  +'alpha={:.2f}'.format(alpha)+'vparameters.csv').reshape((nsubspace,))
#     prob1,nsqz1,ncoh=probability_BIG(Adj,max_clique,c,v,alpha,loop=True)
#     prob2,_=probability_BIG(Adj,max_clique,c,v,alpha,loop=False)
#     ncoh_hist[i]=ncoh
#     problhaf_hist[i]=prob1
#     probhaf_hist[i]=prob2
#     nsqz1_hist[i]=nsqz1
#
# print("alpha",alpha)
# print("Nsqz",nsqz1_hist)
# print("Ratio of displacement",np.divide(ncoh_hist,nsqz1_hist))
# print('Improvement of displacement',np.divide(problhaf_hist,probhaf_hist))
#
#
# hist_lhafnian=[]
# hist_hafnian=[]
# ngraphref=40
# nrandomsubgraphs=20
# edge_density=0.5
# graph_size=np.linspace(10,20,10)
# t0=time()
# prob_haf=[]
# prob_lhaf=[]
# for size in graph_size:
#     tot_lh=0
#     tot_h=0
#     prob_h_temp=0
#     prob_lh_temp=0
#     ngraph_modified=ngraphref
#     for i in range(ngraphref):
#
#         diff_lhaf=np.inf
#         diff_haf=np.inf
#         graph_ref=random_adj_with_loop(int(size),edge_density)
#         max_clique=max_clique_list(graph_ref)
#         if int(np.sum(max_clique))%2==0:
#             lhaf_clique = probability(graph_ref, max_clique, loop=True)
#             haf_clique = probability(graph_ref, max_clique, loop=False)
#
#
#             for j in range(nrandomsubgraphs):
#                 subgraph = random_subgraph_list(graph_ref, len(max_clique))
#                 temp_diff_lh = (lhaf_clique - probability(graph_ref,subgraph, loop=True)).real
#                 temp_diff_h = (haf_clique - probability(graph_ref, subgraph, loop=False)).real
#                 if is_a_clique(subgraph) == False and temp_diff_lh < diff_lhaf:
#                     diff_lhaf = temp_diff_lh
#                 if is_a_clique(subgraph) == False and temp_diff_h < diff_haf:
#                     diff_haf = temp_diff_h
#         else:
#             ngraph_modified-=1
#             diff_haf=0
#             diff_lhaf=0
#             haf_clique=0
#             lhaf_clique=0
#
#         tot_h += diff_haf
#         tot_lh += diff_lhaf
#         prob_h_temp += haf_clique
#         prob_lh_temp += lhaf_clique
#
#
#     hist_hafnian.append(tot_h / ngraph_modified)
#     hist_lhafnian.append(tot_lh / ngraph_modified)
#     prob_haf.append(prob_h_temp/ngraph_modified)
#     prob_lhaf.append(prob_lh_temp/ngraph_modified)
#

#
#
#
# tf=time()
# print('Running time:{:.2f}s'.format(tf-t0))
# fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(16,16))
# ax.plot(graph_size,np.array(hist_lhafnian),label='Loop hafnian',color='g')
# ax.plot(graph_size,np.array(hist_hafnian),label='Hafnian',color='r')
# ax.set_xlabel('Number of nodes of the graph')
# ax.set_ylabel('Minimum distance between the max clique and a non-clique subgraph of same size')
# ax.text(0.1,0.9,r'$N_{graph}$'+'={:.1f}'.format(ngraphref)+'\n'+r'$N_{sub}$'+'={:.1f}'.format(nrandomsubgraphs),transform=ax.transAxes,fontsize=15)
# plt.legend()
# fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(16,16))
# ax.plot(graph_size,np.divide(np.array(hist_lhafnian),np.array(hist_hafnian)),color='r')
# ax.set_xlabel('Number of nodes of the graph')
# ax.set_ylabel('Relative improvement')
# ax.text(0.1,0.9,r'$N_{graph}$'+'={:.1f}'.format(ngraphref)+'\n'+r'$N_{sub}$'+'={:.1f}'.format(nrandomsubgraphs),transform=ax.transAxes,fontsize=15)
# plt.legend()
# plt.show()
# fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(16,16))
# ax.plot(graph_size,np.array(prob_lhaf),label='Displaced GBS',color='g')
# ax.plot(graph_size,np.array(prob_haf),label='GBS',color='r')
# ax.set_xlabel('Number of nodes of the graph')
# ax.set_ylabel('Max clique probability')
# ax.text(0.1,0.9,r'$N_{graph}$'+'={:.1f}'.format(ngraphref),transform=ax.transAxes,fontsize=15)
# plt.legend()
# fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(16,16))
# ax.plot(graph_size,np.divide(np.array(prob_lhaf),np.array(prob_haf)),color='r')
# ax.set_xlabel('Number of nodes of the graph')
# ax.set_ylabel('Ratio of probability lhaf/haf')
# ax.text(0.1,0.9,r'$N_{graph}$'+'={:.1f}'.format(ngraphref),transform=ax.transAxes,fontsize=15)
# plt.legend()
# plt.show()







