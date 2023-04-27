from Library_Generate_displaced_samples_alternative_encoding_TaceAs import *
from Library_Analysis import*
from time import time
from matplotlib import cm
from matplotlib.ticker import IndexLocator
from scipy import ndimage
# cwd='big\\big_tau1.1_.csv'
# BIG=log_data(cwd)
plt.rcParams.update({'font.size':23})
plt.close('all')
nsubspace=24
tau=1.1
alpha=2

start_all=time()
nsamples=10000

nsqz_targ=0.5
ncoh_min=3
ncoh_max=10
ncoh_tot=5
loss_min=0.1
loss_max=0.9
loss_tot=9

n_iterations_local_search=5
loss_mode=np.linspace(loss_min,loss_max,loss_tot)
data_directory = create_directory()

Adj,_=make_adj(tau)

x_target_ncoh=np.linspace(ncoh_min,ncoh_max,ncoh_tot)
y_loss=np.linspace(loss_min,loss_max,loss_tot)
succ_sqzcoh_gbs=np.zeros((loss_tot,ncoh_tot))
succ_sqzcoh_uni=np.zeros((loss_tot,ncoh_tot))

for i in range(len(loss_mode)):
    for j in range(len(x_target_ncoh)):
        samples, path, ncoh, nsqz= samples_cov_alt(Adj=Adj, alpha=alpha, target_nsqz=nsqz_targ, target_ncoh=x_target_ncoh[j],
                                              nsamples=nsamples, data_directory=data_directory, loss_mode=loss_mode[i],
                                              hbar=2, n_subspace=nsubspace)
        time1 = time() - start_all
        print("Time to create the samples in seconds:",time1)

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

        np.savetxt('succ_gbs_ncoh_TaceAs={:.1f}_nsqz={:.1f}_nsamples={:.1f},loss_min={:.1f},loss_max={:.1f},ncoh_min={:.1f},ncoh_max={:.1f},niteration={:.1f}.txt'.format(ncoh,nsqz_targ,nsamples,loss_min,loss_max,ncoh_min,ncoh_max,n_iterations_local_search), np.array(succ_gbs), delimiter=',')
        np.savetxt('succ_uni_ncoh_TaceAs={:.1f}_nsqz={:.1f}_nsamples={:.1f},loss_min={:.1f},loss_max={:.1f},ncoh_min={:.1f},ncoh_max={:.1f},niteration={:.1f}.txt'.format(ncoh,nsqz_targ,nsamples,loss_min,loss_max,ncoh_min,ncoh_max,n_iterations_local_search),np.array(succ_uni),delimiter=',')

        print("ITERATION:i={:.1f},j={:.1f}".format(i,j))

plt.close('all')

succ_sqzcoh_gbs=succ_sqzcoh_gbs/100
succ_sqzcoh_uni=succ_sqzcoh_uni/100

np.savetxt('Array_succ_gbs_nsamples_TaceAs={:.1f},nsqz_min={:.1f},nsqz_max={:.1f},ncoh_min={:.1f},ncoh_max={:.1f},nsqz={:.1f},niteration={:.1f}.txt'.format(nsamples,loss_min,loss_max,ncoh_min,ncoh_max,nsqz,n_iterations_local_search),succ_sqzcoh_gbs,delimiter=',')
np.savetxt('Array_succ_uni_nsamples_TaceAs={:.1f},nsqz_min={:.1f},nsqz_max={:.1f},ncoh_min={:.1f},ncoh_max={:.1f},nsqz={:.1f},niteration={:.1f}.txt'.format(nsamples,loss_min,loss_max,ncoh_min,ncoh_max,nsqz,n_iterations_local_search),succ_sqzcoh_uni,delimiter=',')

fig=plt.figure(figsize=plt.figaspect(0.4))
ax=fig.add_subplot(1,2,1)
array_gbs=ax.imshow(succ_sqzcoh_gbs,cmap=cm.viridis,extent=[ncoh_min,ncoh_max,loss_max,loss_min])
fig.colorbar(array_gbs,shrink=0.5)
ax.set_title('Success rate vs loss \n and displacement with a displaced GBS sampler \nafter {:.1f} iterations and sinhr**2={:.1f}'.format(n_iterations_local_search,nsqz_targ) )
ax.set_xlabel(r'$\langle n_{coh}\rangle$')
ax.set_ylabel(r'$Loss$')
ax=fig.add_subplot(1,2,2)
array_uni=ax.imshow(succ_sqzcoh_uni,cmap=cm.viridis,extent=[ncoh_min,ncoh_max,loss_max,loss_min])
fig.colorbar(array_uni,shrink=0.5)
ax.set_title('Success rate vs loss\n and displacement with a uniform sampler after \n{:.1f} iterations and sinhr**2={:.1f}'.format(n_iterations_local_search,nsqz_targ) )
ax.set_xlabel(r'$\langle n_{coh}\rangle$')
ax.set_ylabel(r'$Loss$')
plt.tight_layout()
plt.savefig('Success Rate vs Loss and Displacement TaceAS,nsamples={:.1f},loss_min={:.1f},loss_max={:.1f},ncoh_min={:.1f},ncoh_max={:.1f},nsqz={:.1f},niteration={:.1f}.pdf'.format(nsamples,loss_min,loss_max,ncoh_min,ncoh_max,nsqz_targ,n_iterations_local_search),format='pdf')
fig.show()
plt.pause(10000)
