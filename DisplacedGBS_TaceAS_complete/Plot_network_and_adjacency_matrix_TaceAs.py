import copy

import matplotlib.pyplot as plt
import numpy as np

from Library_Generate_displaced_samples_alternative_encoding_TaceAs import *
from Library_Analysis import*
from time import time
from matplotlib import cm
from matplotlib.ticker import FixedLocator
# cwd='big\\big_tau1.1_.csv'
# BIG=log_data(cwd)

plt.close('all')
nsubspace=9
tau=1.1
alpha=1

start_all=time()
Adj,_=make_adj(tau)
Adj_weighted=copy.deepcopy(Adj)
weights=make_potential_vect()[:nsubspace]
for i in range(len(Adj)):
    for j in range(len(Adj[0])):
        Adj_weighted[i,j]=Adj[i,j]*weights[i]*weights[j]

plt.close('all')
print(Adj_weighted)
fig=plt.figure(figsize=plt.figaspect(0.4))

ax=fig.add_subplot(1,2,1)

ax.xaxis.set_major_locator(FixedLocator([0,8]))
ax.yaxis.set_major_locator(FixedLocator([0,8]))
ax.set_title('Adjacency matrix')
ax.set_xlabel(r'$i$')
ax.set_ylabel(r'$j$')
im=ax.imshow(np.abs(Adj_weighted),cmap=cm.Blues)
fig.colorbar(im,ax=ax,label=r'$\bf A_{adj}$',shrink=0.5)


ax=fig.add_subplot(1,2,2)
graph_ref = nx.Graph(Adj_weighted)
nx.draw_networkx(graph_ref)
ax.set_title('4uxb graph' )
plt.tight_layout()
plt.savefig('Plot_graph.png',format='png')
plt.savefig('Plot_graph.svg',format='svg')
fig.show()
plt.pause(1000)
