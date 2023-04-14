import matplotlib.pyplot as plt
import numpy as np
from LoopHafnian import*
import networkx as nx
sq_min=0.2
sq_max=0.4
nsqz=0.5
nsubspace=24
alpha=1.5
cwd = 'big\\adj_mat_tau1.1_.csv'
Adj= log_data(cwd)
target_ncoh=1
weights=make_potential_vect()[:len(Adj)]
graph=nx.from_numpy_array(Adj)

cliques=list(nx.find_cliques(graph))
cliques_n=[]
for i in range (len(cliques)):
    if len(cliques[i])==8:
        clique=[0 for i in range(24)]
        for el in cliques[i]:
            clique[el]=1
        cliques_n.append(clique)


def sample_weight(sample,weight):
    #Given a sample and the list of weights per nodes, return the weight of the sample
    #sample=a 1D array for the sample. WARNING: Collision-free regime assumed (only 0 or 1 in the sample) and click convention
    # list of weights used to build the BIG matrix with the same length as the sample
    return np.sum(np.multiply(sample,weight))

c = log_data('Parameters_c_v\\TaceAs\\' + 'sqmin={:.1f}'.format(sq_min) + 'sqmax={:.1f}'.format(sq_max) + 'dim={:.1f}'.format(nsubspace) + 'ncoh={:.3f}'.format(target_ncoh) + 'alpha={:.2f}'.format(alpha) + 'cparameters.csv').reshape((nsubspace,))
v = log_data('Parameters_c_v\\TaceAs\\' + 'sqmin={:.1f}'.format(sq_min) + 'sqmax={:.1f}'.format(sq_max) + 'dim={:.1f}'.format(nsubspace) + 'ncoh={:.3f}'.format(target_ncoh) + 'alpha={:.2f}'.format(alpha) + 'vparameters.csv').reshape((nsubspace,))
weigths= make_potential_vect()
max_clique=max_clique_list(Adj,weigths)
prob_max_clique_loop,d_alpha=probability_c(Adj, max_clique,alpha,target_nsqz=nsqz,loop=True)
ncoh=np.sum(d_alpha**2)
prob_max_clique=probability_c(Adj, max_clique,alpha,target_nsqz=nsqz+ncoh,loop=False)
# prob_max_clique_loop,_,_=probability_BIG(Adj, max_clique, c, v, alpha, loop=True)
# prob_max_clique,_=probability_BIG(Adj, max_clique, c, v, alpha, loop=False)
hist_prob_cliques_loop=[]
hist_prob_cliques=[]
for el in cliques_n:
    # prob1,_,_= probability_BIG(Adj,el, c, v, alpha, loop=True)
    # prob2, _ = probability_BIG(Adj, el, c, v, alpha, loop=False)
    prob1,d_alpha = probability_c(Adj, el,alpha,nsqz, loop=True)
    prob2= probability_c(Adj, el,alpha,nsqz+ncoh, loop=False)
    hist_prob_cliques_loop.append(prob1)
    hist_prob_cliques.append(prob2)
print("d_alpha",np.sum(d_alpha**2))
x=np.linspace(0,len(hist_prob_cliques_loop),len(hist_prob_cliques_loop))
fig,ax=plt.subplots(nrows=1,ncols=2)
bars1=ax[0].bar(x=x,height=hist_prob_cliques_loop,color='g')
labels=[sample_weight(el,weights) for el in cliques_n]
ax[0].bar_label(bars1,labels=["{:.2f}".format(el) for el in labels])
bars2=ax[1].bar(x=x,height=hist_prob_cliques,color='r')
ax[1].bar_label(bars2,labels=["{:.2f}".format(el) for el in labels])
fig.suptitle('Histograms for the max clique and the other cliques of same size for displaced GBS with nsqz={:.1f} and ncoh={:.1f}(left) and normal GBS case (right) with nsqz={:.1f} '.format(nsqz,ncoh,ncoh+nsqz))
for a in ax.flat:
    a.set(ylabel='Probability')
plt.show()




