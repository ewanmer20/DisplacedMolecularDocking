import matplotlib.pyplot as plt
import numpy as np

from Library_sampling_displaced_GBS_4uxb import *
from Library_samples_analysis import*
from time import time
from matplotlib import cm
from matplotlib.ticker import IndexLocator
# cwd='big\\big_tau1.1_.csv'
# BIG=log_data(cwd)

plt.close('all')
nsubspace=9
# Adj = data.TaceAs().adj[:nsubspace,:nsubspace]
tau=1.1
alpha=1
data_directory=create_directory()
Adj,_=make_adj(tau)
start_all=time()
nsamples=10000
target_nsqz=3
target_ncoh=1.5
n_loss_points=10
loss_mode=np.linspace(0,0.9,n_loss_points)
n_iterations_local_search=5
succ_loss_gbs=np.zeros((n_loss_points,n_iterations_local_search))
succ_loss_uni=np.zeros((n_loss_points,n_iterations_local_search))
for i in range(n_loss_points):
    samples,path,ncoh,nsqz= samples_cov_alt(Adj=Adj,alpha=alpha,target_nsqz=target_nsqz,target_ncoh=target_ncoh,nsamples=nsamples,data_directory=data_directory,loss_mode=loss_mode[i],hbar=2,n_subspace=nsubspace)
    time1 = time() - start_all
    print("Time to create the samples in seconds:",time1)
    #Using the adjacency matrix reduced to the first 10 modes to be consistent with the GBS simulation
    # Adj =data.TaceAs().adj

    #Retrieving the potential values for the adjacency matrix
    weights=make_potential_vect()

    # Histogram of the number of photons per sample
    a,nmax=plot_histogram(samples,plot=False)
    print(a)

    # #Remove the non-zero clicks event and the collision events (just for PNRS detectors)
    cleaned_samples,ncollision=clean_samples(samples,nmax)
    print(len(cleaned_samples))
    print("Number of collisions:",ncollision)

    #Find the maximum clique with classical algorithm

    clique_max,clique_weight=find_max_clique(Adj,weights,networkx_conv=False)
    print('Max clique',clique_max)
    print('Weights of the max clique',clique_weight)

    searched_gbs,searched_uni=plot_success_rate_vs_niter(cleaned_GBS_samples=cleaned_samples,Adj=Adj,weights=weights,niter=n_iterations_local_search,plot=False)
    plot_histogram_clique_values(cleaned_GBS_samples=cleaned_samples,nmax=nmax,Adj=Adj,weights=weights,plot=False)

    succ_loss_gbs[i, :] = searched_gbs
    succ_loss_uni[i, :] = searched_uni



    print("ITERATION:{:.1f}".format(i))

np.savetxt('succ_loss_gbs.txt',np.array(succ_loss_gbs),delimiter=',')
np.savetxt('succ_loss_uni.txt',np.array(succ_loss_uni),delimiter=',')

plt.close('all')
succ_loss_gbs=succ_loss_gbs/100
succ_loss_uni=succ_loss_uni/100
fig=plt.figure(figsize=plt.figaspect(0.4))

Iterations,Loss=np.meshgrid(np.linspace(1,n_iterations_local_search,n_iterations_local_search),loss_mode)

ax=fig.add_subplot(1,2,1,projection='3d')
surf_gbs=ax.plot_surface(Loss,Iterations,succ_loss_gbs,cmap=cm.viridis)
ax.zaxis.set_major_locator(IndexLocator(base=0.1,offset=0))
fig.colorbar(surf_gbs,shrink=0.5)
ax.set_title('Success rate vs loss \n and number of iterations with a GBS sampler' )
ax.set_zlim(-0.1,1.1)
ax.set_xlabel(r'$Loss$')
ax.set_ylabel(r'$Number of iterations of local search$')

ax=fig.add_subplot(1,2,2,projection='3d')
surf_uni=ax.plot_surface(Loss,Iterations,succ_loss_uni,cmap=cm.viridis)
fig.colorbar(surf_uni,shrink=0.5)
ax.zaxis.set_major_locator(IndexLocator(base=0.1,offset=0))
ax.set_title('Success rate vs loss \n and number of iterations with a uniform sampler' )
ax.set_zlim(-0.1,1.1)
ax.set_xlabel(r'$Loss$')
ax.set_ylabel(r'$Number of iterations of local search$')


plt.tight_layout()
plt.savefig('Success Rate vs Iteration And Loss,nsqz={:.2f}ncoh={:.2f}.svg'.format(nsqz,ncoh),format='svg',transparent=True)
fig.show()
