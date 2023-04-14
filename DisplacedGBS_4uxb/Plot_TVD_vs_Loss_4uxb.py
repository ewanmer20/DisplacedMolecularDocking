import copy

import matplotlib.pyplot as plt
from time import time

import numpy as np
from matplotlib.ticker import IndexLocator,LinearLocator,FixedLocator
from Generate_displaced_samples_alternative_encoding_4uxb import *
import os
plt.rcParams['font.size']=12
plt.close('all')
n_subspace=9
tau=1.1
alpha=1
data_directory=os.getcwd()
start_all=time()
Adj,_=make_adj(tau)
target_ncoh=np.array([3,4])
target_nsqz=np.array([0.1,0.5,1,2,3])
cutoff=4 # Finite dimension of the photon number fock space of each mode
n_loss_params=50
nsamples=10000
loss_uni=np.ones(n_subspace)*0.5 #Uniform loss for each mode
loss_params=np.linspace(0.1,0.9,n_loss_params)

tvd_array1=np.zeros((len(target_nsqz),n_loss_params))
tvd_array2=np.zeros((len(target_nsqz),n_loss_params))

#Upper and lower bound given by TheWalrus caused by the truncation of the Fock space
bound1=np.zeros((len(target_nsqz),n_loss_params))
bound2=np.zeros((len(target_nsqz),n_loss_params))




array_index=generate_threefoldstatistics(n_subspace,cutoff)
for l in range(len(target_ncoh)):
        for i in range(len(target_nsqz)):

            # Calculate the covariance matrix and the displacement vector given the target number of squeezed states and
            # displacements and the adjacency matrix in the real convention (xxpp convention used by the Walrus)
            c = tune_c(alpha, target_nsqz[i], Adj, n_subspace)
            cov, mean = create_cov_mean_alt(Adj, c, alpha, target_ncoh[l], n_subspace, conv='real')
            # Calculate the probability distribution of the output gaussian states in the Fock number basis
            probabilities_lossfree = qt.probabilities(mu=mean, cov=cov, cutoff=cutoff)

            # Reshape the probabilities tensor to one dimensional arrays

            probabilities_lossfree_reshaped=copy.deepcopy(probabilities_lossfree)
            probabilities_lossfree_reshaped = np.reshape(probabilities_lossfree_reshaped, (cutoff ** (n_subspace)))

            # Bounds of truncated TVD
            bound_lossfree = qt.tvd_cutoff_bounds(mu=mean, cov=cov, cutoff=cutoff)[-1]


            for j in range(len(loss_params)):

                #Calculate the covariance matrix and the displacement vector given the target number of squeezed states and displacements of the loss case

                mu_loss = copy.deepcopy(mean)
                cov_loss = copy.deepcopy(cov)
                for k in range(n_subspace):
                    mu_loss, cov_loss = loss(mu=mu_loss, cov=cov_loss, T=1-loss_params[j], nbar=0, mode=k)


                #Update the probability distribution to take into account the loss, assuming all the losses are located just before the detectors and are uniform
                loss_array = np.ones(n_subspace) * loss_params[j]
                probabilities_withloss=qt.update_probabilities_with_loss(etas=1-loss_array,probs=probabilities_lossfree)
                # Reshape the probabilities tensor to one dimensional arrays
                probabilities_withloss = np.reshape(probabilities_withloss, (cutoff ** (n_subspace)))


                #Bounds of truncated TVDs
                bound_loss=qt.tvd_cutoff_bounds(mu=mu_loss,cov=cov_loss,cutoff=cutoff)[-1]
                bound=np.sqrt(bound_loss**2+bound_lossfree**2)
                if l==0:
                    bound1[i,j]=bound
                    tvd_array1[i,j]=tvd(probabilities_lossfree_reshaped,probabilities_withloss)
                else:
                    bound2[i, j] = bound
                    tvd_array2[i, j] = tvd(probabilities_lossfree_reshaped, probabilities_withloss)
                if j==10 and l==0:
                    select_threefoldstatistics(probability_tensor_groundthruth=probabilities_lossfree_reshaped,probability_tensor_experiment=probabilities_withloss,array_index=array_index,numodes=n_subspace,cutoff=cutoff,file_title='Histogram_target_ncoh={:.1f}tarhet_nsqz={:.1f}loss={:.1f}TVD{:.4f}'.format(target_ncoh[l],target_nsqz[i],loss_params[j],tvd_array1[i,j]))

                print("ITERATION:i={:.1f},j={:.1f}".format(i, j))




r=np.arcsinh(np.sqrt(target_nsqz)) # Conversion to n_sqz to r for the rendering
fig=plt.figure(figsize=plt.figaspect(0.4))
ax=fig.add_subplot(121)
for i in range(len(target_nsqz)):
    ax.plot(loss_params,tvd_array1[i,:],label=r'$\langle n\rangle$={:.1f}'.format(target_nsqz[i]))
    ax.fill_between(loss_params,tvd_array1[i,:]-bound1[i,:],tvd_array1[i,:]+bound1[i,:],alpha=0.2)
ax.yaxis.set_major_locator(LinearLocator(numticks=5))
ax.xaxis.set_major_locator(IndexLocator(base=0.1,offset=0))
ax.legend()
ax.set_xlabel('Loss, '+r'$\eta$')
ax.set_ylabel('TVD')



ax=fig.add_subplot(122)
for i in range(len(target_nsqz)):
    ax.plot(loss_params,tvd_array2[i,:],label=r'$\langle n\rangle$={:.1f}'.format(target_nsqz[i]))
    ax.fill_between(loss_params, tvd_array2[i, :]-bound2[i,:], tvd_array2[i, :] + bound2[i,:], alpha=0.2)

ax.yaxis.set_major_locator(LinearLocator(numticks=5))
ax.xaxis.set_major_locator(IndexLocator(base=0.1,offset=0))
ax.legend()
ax.set_xlabel('Loss, '+r'$\eta$')
ax.set_ylabel('TVD')
fig.suptitle(r'Total variation distribution (TVD) of the normalized probability distribution for ncoh={:.1f}(Left),{:.1f}(Right) and cutoff={:d} as a function of loss $\eta$ and squeezing parameter r'.format(target_ncoh[0],target_ncoh[1],cutoff),wrap=True)
plt.savefig('Total variation distribution for ncoh={:.1f},{:.1f} as a function of loss.pdf'.format(target_ncoh[0],target_ncoh[1]),format='pdf')
plt.savefig('Total variation distribution for ncoh={:.1f},{:.1f} as a function of loss.png'.format(target_ncoh[0],target_ncoh[1]),format='png')
fig.show()
plt.pause(200)




