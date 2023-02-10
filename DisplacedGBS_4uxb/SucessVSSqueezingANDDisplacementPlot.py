import matplotlib.pyplot as plt
import numpy as np

from Generate_displaced_samples_alternative_encoding_4uxb import *
from Analysis_lib import*
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

start_all=time()
nsamples=10000

n_iterations_local_search=7
loss_mode=0.5
data_directory = create_directory()

Adj,_=make_adj(tau)
target_ncoh=np.linspace(3,10,5)
target_nsqz=np.linspace(0.1,2,10)
succ_sqzcoh_gbs=np.zeros((len(target_ncoh),len(target_nsqz)))
succ_sqzcoh_uni=np.zeros((len(target_ncoh),len(target_nsqz)))
n_iterations_local_search=1
for i in range(len(target_ncoh)):
    for j in range(len(target_nsqz)):
        samples, path, ncoh, nsqz= samples_cov_alt(Adj=Adj, alpha=alpha, target_nsqz=target_nsqz[j], target_ncoh=target_ncoh[i],
                                              nsamples=nsamples, data_directory=data_directory, loss_mode=loss_mode,
                                              hbar=2, n_subspace=nsubspace)
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

        succ_gbs,succ_uni=plot_success_rate_vs_niter(cleaned_GBS_samples=cleaned_samples,Adj=Adj,weights=weights,niter=n_iterations_local_search,plot=False)
        plot_histogram_clique_values(cleaned_GBS_samples=cleaned_samples,nmax=nmax,Adj=Adj,weights=weights,plot=False)

        succ_sqzcoh_gbs[i,j]=succ_gbs[-1]
        succ_sqzcoh_uni[i, j] = succ_uni[-1]

        np.savetxt('succ_gbs_ncoh={:.1f}_nsqz={:.1f}.txt'.format(ncoh,nsqz), np.array(succ_gbs), delimiter=',')
        np.savetxt('succ_uni_ncoh={:.1f}_nsqz={:.1f}.txt'.format(ncoh,nsqz),np.array(succ_uni),delimiter=',')

        print("ITERATION:i={:.1f},j={:.1f}".format(i,j))

plt.close('all')

succ_sqzcoh_gbs=succ_sqzcoh_gbs/100
succ_sqzcoh_uni=succ_sqzcoh_uni/100

fig=plt.figure(figsize=plt.figaspect(0.4))
Ncoh,Nsqz=np.meshgrid(target_ncoh,target_nsqz)
ax=fig.add_subplot(1,2,1,projection='3d')
surf_gbs=ax.plot_surface(Ncoh,Nsqz,succ_sqzcoh_gbs,cmap=cm.viridis)
ax.zaxis.set_major_locator(IndexLocator(base=1,offset=0))
fig.colorbar(surf_gbs,shrink=0.5)
ax.view_init(elev=23,azim=-69)
ax.set_title('Success rate vs the mean photon number of squeezing\n and displacement with a GBS displaced sampler' )
# ax.set_zlim(-0.1,1.1)
ax.set_xlabel(r'$\langle n_{coh}\rangle$')
ax.set_ylabel(r'$\langle n_{sqz}\rangle$')
ax.set_zlabel('Sucess rate after {:.1f} iterations'.format(n_iterations_local_search))

ax=fig.add_subplot(1,2,2,projection='3d')
surf_uni=ax.plot_surface(Ncoh,Nsqz,succ_sqzcoh_gbs,cmap=cm.viridis)
ax.zaxis.set_major_locator(IndexLocator(base=1,offset=0))
fig.colorbar(surf_uni,shrink=0.5)
ax.view_init(elev=23,azim=-69)
ax.set_title('Success rate vs the mean photon number of squeezing\n and displacement with a uniform sampler' )
# ax.set_zlim(-0.1,1.1)
ax.set_xlabel(r'$\langle n_{coh}\rangle$')
ax.set_ylabel(r'$\langle n_{sqz}\rangle$')
ax.set_zlabel('Sucess rate after {:.1f} iterations'.format(n_iterations_local_search))



plt.tight_layout()
plt.savefig('Success Rate vs Iteration And Loss,nsqz={:.2f}ncoh={:.2f}.svg'.format(nsqz,ncoh),format='svg',transparent=True)
fig.show()
plt.pause(100)

