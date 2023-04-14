import copy

import matplotlib.pyplot as plt
from time import time

import numpy as np
from matplotlib.ticker import IndexLocator,LinearLocator,FixedLocator
from Generate_displaced_samples_alternative_encoding_4uxb import *
import os
plt.rcParams['font.size']=12
def tvd(prob1,prob2):
    """

    :param prob1: A one-dimensional array containing the different probabilities of the first distribution
    :param prob2: A one-dimensional array containing the different probabilities of the same length of the first probability
    :return: the Total variation distance between the two renormalized probability distributions prob1 and prob2
    """
    if len(prob1)==len(prob2):
        prob1_copy=copy.deepcopy(prob1)/(np.sum(prob1))
        prob2_copy=copy.deepcopy(prob2)/(np.sum(prob2))

        return 0.5*np.sum(np.abs(prob1_copy-prob2_copy))
    else:
        print("prob1 and prob2 have to be the same length!")





plt.close('all')
n_subspace=9
tau=1.1
alpha=1
data_directory=os.getcwd()
start_all=time()
Adj,_=make_adj(tau)
target_ncoh=np.array([3,4])
target_nsqz=np.array([0.1,1,2])
cutoff=4 # Finite dimension of the photon number fock space of each mode
n_noise_params=10
n_samples_noise=100 #Number of samples with the same level noise taken to average the tvd

noise_uni=np.ones(n_subspace)*0.5 #Uniform noise for each mode
noise_params=np.linspace(0.05,1,n_noise_params)

tvd_array1=np.zeros((len(target_nsqz),n_noise_params))
tvd_array2=np.zeros((len(target_nsqz),n_noise_params))

#Upper and lower bound given by TheWalrus caused by the truncation of the Fock space
bound1=np.zeros((len(target_nsqz),n_noise_params))
bound2=np.zeros((len(target_nsqz),n_noise_params))


#First mean displacement
for l in range(len(target_ncoh)):
    for i in range(len(target_nsqz)):

        # Calculate the covariance matrix and the displacement vector given the target number of squeezed states and
        # displacements and the adjacency matrix in the real convention (xxpp convention used by the Walrus)
        c = tune_c(alpha, target_nsqz[i], Adj, n_subspace)
        cov, mean = create_cov_mean_alt(Adj, c, alpha, target_ncoh[l], n_subspace, conv='real')
        # Calculate the probability distribution of the output gaussian states in the Fock number basis
        probabilities_noisefree = qt.probabilities(mu=mean, cov=cov, cutoff=cutoff)

        # Reshape the probabilities tensor to one dimensional arrays

        probabilities_noisefree_reshaped=copy.deepcopy(probabilities_noisefree)
        probabilities_noisefree_reshaped = np.reshape(probabilities_noisefree_reshaped, (cutoff ** (n_subspace)))

        # Bounds of truncated TVD
        bound_noisefree = qt.tvd_cutoff_bounds(mu=mean, cov=cov, cutoff=cutoff)[-1]


        for j in range(len(noise_params)):

            tvd_temp=np.zeros(n_samples_noise)
            for k in range(n_samples_noise):
                mean_noise=np.zeros(len(mean),dtype=np.complex64)

                for m in range(len(mean)):
                    mean_noise[m]=mean[m]*np.exp(1j*np.random.normal(loc=0,scale=noise_params[j]/np.pi))

                #Update the probability distribution to take into account the noise, assuming all the noise is located just before the detectors and are uniform
                probabilities_withnoise=qt.probabilities(mu=mean_noise, cov=cov, cutoff=cutoff)

                #Reshape the probabilities tensor to one dimensional arrays
                probabilities_withnoise=np.reshape(probabilities_withnoise,(cutoff**(n_subspace)))

                tvd_temp[k] = tvd(probabilities_noisefree_reshaped, probabilities_withnoise)

            #Bounds of truncated TVDs
            if l==0:
                bound1[i,j]=np.sqrt(bound_noisefree**2)
                tvd_array1[i,j]=np.sum(tvd_temp)/n_samples_noise
            if l==1:
                bound2[i, j] = np.sqrt(bound_noisefree ** 2)
                tvd_array2[i, j] = np.sum(tvd_temp) / n_samples_noise
            print("ITERATION:l={:.1f},i={:.1f},j={:.1f}".format(l,i, j))






r=np.arcsinh(np.sqrt(target_nsqz)) # Conversion to n_sqz to r for the rendering
fig=plt.figure(figsize=plt.figaspect(0.4))
ax=fig.add_subplot(121)
for i in range(len(target_nsqz)):
    ax.plot(noise_params,tvd_array1[i,:],label=r'$\langle n\rangle$={:.1f}'.format(target_nsqz[i]))
    # ax.fill_between(noise_params,tvd_array1[i,:]-bound1[i,:],tvd_array1[i,:]+bound1[i,:],alpha=0.2)
ax.yaxis.set_major_locator(LinearLocator(numticks=5))
ax.xaxis.set_major_locator(IndexLocator(base=0.1,offset=0))
ax.legend()
ax.set_xlabel('Noise standard deviation, '+r'$\sigma/\pi$')
ax.set_ylabel('TVD')



ax=fig.add_subplot(122)
for i in range(len(target_nsqz)):
    ax.plot(noise_params,tvd_array2[i,:],label=r'$\langle n\rangle$={:.1f}'.format(target_nsqz[i]))
    # ax.fill_between(noise_params, tvd_array2[i, :]-bound2[i,:], tvd_array2[i, :] + bound2[i,:], alpha=0.2)

ax.yaxis.set_major_locator(LinearLocator(numticks=5))
ax.xaxis.set_major_locator(IndexLocator(base=0.1,offset=0))
ax.legend()
ax.set_xlabel('Noise standard deviation, '+r'$\sigma/\pi$')
ax.set_ylabel('TVD')
fig.suptitle(r'Total variation distribution (TVD) of the normalized probability distribution for ncoh={:.1f}(Left),{:.1f}(Right) and cutoff={:d} as a function of noise $\sigma$ and squeezing parameter r'.format(target_ncoh[0],target_ncoh[1],cutoff),wrap=True)
plt.savefig('Total variation distribution for ncoh={:.1f},{:.1f} as a function of noise.pdf'.format(target_ncoh[0],target_ncoh[1]),format='pdf')
plt.savefig('Total variation distribution for ncoh={:.1f},{:.1f} as a function of noise.png'.format(target_ncoh[0],target_ncoh[1]),format='png')
fig.show()
plt.pause(200)

