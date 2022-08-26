import matplotlib.pyplot as plt
import numpy as np

from Analysis_lib import*
from Generate_samples import*
from time import time
import thewalrus as thw
import random
import networkx as nx

def random_subgraph(adj,nvertices):
    indexes=np.array(random.sample(range(len(adj)),nvertices)).astype(np.int64)
    seq=np.zeros(len(adj),dtype=np.int64)
    for i in range(len(indexes)):
        seq[indexes[i]]=1
    return thw.reduction(adj,seq)

def max_clique_graph(adj,weights=None):
    if weights is None:
        weights=np.ones(len(adj))
    clique_max_seq, clique_weight = find_max_clique(adj, weights, networkx_conv=False)
    max_clique = thw.reduction(adj, clique_max_seq.astype(np.int64))
    return max_clique

def random_adj(nvertices,prob_edges):
    graph=nx.erdos_renyi_graph(n=nvertices,p=prob_edges,directed=False)
    adj=nx.to_numpy_array(graph)
    for i in range(len(adj)):
        adj[i,i]=1
    return adj
def is_a_clique(adj):
    if np.sum(adj)==len(adj)**2:
        return True
    else:
        return False

cwd = 'big\\adj_mat_tau1.1_.csv'
BIG = log_data(cwd)
for i in range(len(BIG)):
    BIG[i,i]=1

A=random_adj(10,0.2)
subgraph=random_subgraph(BIG,8)
weights = make_potential_vect()
max_clique=max_clique_graph(BIG,weights=weights)
max_clique_rand=max_clique_graph(adj=A)

hist_lhafnian=[]
hist_hafnian=[]
ngraphref=500
nrandomsubgraphs=1000
edge_density=0.5
graph_size=np.linspace(5,20,16)
t0=time()
for size in graph_size:
    tot_lh=0
    tot_h=0
    for i in range(ngraphref):
        diff_lhaf=np.inf
        diff_haf=np.inf
        graph_ref=random_adj(int(size),edge_density)
        max_clique=max_clique_graph(graph_ref)
        lhaf_clique=thw.hafnian(max_clique, loop=True)
        haf_clique=thw.hafnian(max_clique)
        for j in range(nrandomsubgraphs):
            subgraph=random_subgraph(graph_ref,len(max_clique))
            temp_diff_lh= (lhaf_clique - thw.hafnian(subgraph, loop=True)).real
            temp_diff_h= (haf_clique - thw.hafnian(subgraph)).real
            if is_a_clique(subgraph)==False and temp_diff_lh<diff_lhaf:
                diff_lhaf=temp_diff_lh
            if is_a_clique(subgraph)==False and temp_diff_h<diff_haf:
                diff_haf=temp_diff_h
        tot_h+=diff_haf
        tot_lh+=diff_lhaf
    hist_hafnian.append(tot_h/ngraphref)
    hist_lhafnian.append(tot_lh/ngraphref)
tf=time()
print('Running time:{:.2f}s'.format(tf-t0))
fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(16,16))
ax.plot(graph_size,np.array(hist_lhafnian),label='Loop hafnian',color='g')
ax.plot(graph_size,np.array(hist_hafnian),label='Hafnian',color='r')
ax.set_xlabel('Number of nodes of the graph')
ax.set_ylabel('Minimum distance between the max clique and a non-clique subgraph of same size')
ax.text(0.1,0.9,r'$N_{graph}$'+'={:.1f}'.format(ngraphref)+'\n'+r'$N_{sub}$'+'={:.1f}'.format(nrandomsubgraphs),transform=ax.transAxes,fontsize=15)
plt.legend()
fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(16,16))
ax.plot(graph_size,np.divide(np.array(hist_lhafnian),np.array(hist_hafnian)),color='r')
ax.set_xlabel('Number of nodes of the graph')
ax.set_ylabel('Relative improvement')
ax.text(0.1,0.9,r'$N_{graph}$'+'={:.1f}'.format(ngraphref)+'\n'+r'$N_{sub}$'+'={:.1f}'.format(nrandomsubgraphs),transform=ax.transAxes,fontsize=15)
plt.legend()

plt.show()







