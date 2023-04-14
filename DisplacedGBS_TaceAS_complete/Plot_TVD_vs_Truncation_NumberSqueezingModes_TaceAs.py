import copy

import matplotlib.pyplot as plt
from time import time

import numpy as np
from matplotlib.ticker import IndexLocator,LinearLocator,FixedLocator
from Library_Generate_displaced_samples_alternative_encoding_TaceAs import *
import os

plt.rcParams['font.size']=12
plt.close('all')
nsubspace=24
tau=1.1
alpha=1
data_directory=os.getcwd()
start_all=time()
Adj = data.TaceAs().adj[:nsubspace,:nsubspace]
target_ncoh=np.array(3)
target_nsqz=np.array([0.1,0.5,1])
num_truncatedmodes=np.array([1,2,3,4,5,6,7,8,9])
tvd_array=np.zeros((len(target_nsqz),len(num_truncatedmodes)))
tvd_bound=np.zeros((len(target_nsqz),len(num_truncatedmodes)))
cutoff=2
hbar=2

array_index=generate_threefoldstatistics(nsubspace,cutoff)

for i in range(len(target_nsqz)):
    # Generation of the probability distriubtion encoding 4uxb graph given ncoh and nsqz without any truncation
    c = tune_c(alpha, target_nsqz[i], Adj, nsubspace)
    Id = np.eye(nsubspace)
    weights = make_potential_vect()[:nsubspace]
    Adj = Adj[:nsubspace, :nsubspace]
    omega = make_omega(c, alpha)[:nsubspace, :nsubspace]
    BIG = omega @ Adj @ omega
    Sigma_Qinv = np.block([[Id, -BIG], [-BIG, Id]])
    Sigma_Q = inv(Sigma_Qinv)
    params = optimize_displacement(Adjtot=Adj, target_ncoh=target_ncoh, omega=omega, weights=weights,nsubspace=nsubspace, hbar=hbar).x
    gamma = give_gamma(kappa=params[0], delta=params[1], omega=omega, weights=weights, nsubspace=nsubspace)
    d_alpha = (Sigma_Q @ gamma)[:nsubspace]
    cov = qt.Covmat(Sigma_Q, hbar=hbar)
    mean = np.sqrt(2 * hbar) * np.concatenate([d_alpha, np.zeros(nsubspace)])
    probabilities_lossfree_full = qt.probabilities(mu=mean, cov=cov, cutoff=cutoff, hbar=hbar)
    # Reshaping of the two probability distributions up to a cutoff photons per mode
    probabilities_lossfree_full_reshaped = copy.deepcopy(probabilities_lossfree_full)

    probabilities_lossfree_full_reshaped = np.reshape(probabilities_lossfree_full_reshaped, (cutoff ** (nsubspace)))



    # Bounds of truncated TVDs
    bound_full = qt.tvd_cutoff_bounds(mu=mean, cov=cov, cutoff=cutoff)[-1]


    for j in range(len(num_truncatedmodes)):

        #Generation of the probability distrtibution but with truncation of the squeezers to remove the n_truncatedmodes weakest squeezers.
        #Gamma and c remains unchanged (an idea could be to reoptimise gamma and see if it is possible to reduce the tvd!)

        (lambdal,U)=takagi(BIG)
        if num_truncatedmodes[j]>=1:
            lambdal[len(lambdal)-num_truncatedmodes[j]:len(lambdal)]=np.zeros(num_truncatedmodes[j])
        BIG_updated=U@np.diag(lambdal)@np.transpose(U)
        Sigma_Qinv_truncated = np.block([[Id, -BIG_updated], [-BIG_updated, Id]])
        Sigma_Q_truncated = inv(Sigma_Qinv_truncated)
        d_alpha_truncated=(Sigma_Q_truncated @ gamma)[:nsubspace]
        cov_truncated=qt.Covmat(Sigma_Q_truncated,hbar=hbar)
        mean_truncated=np.sqrt(2*hbar)*np.concatenate([d_alpha, np.zeros(nsubspace)])
        probabilities_lossfree_truncated = qt.probabilities(mu=mean_truncated, cov=cov_truncated, cutoff=cutoff,hbar=hbar)

        # Reshaping of the two probability distributions up to a cutoff photons per mode
        probabilities_lossfree_truncated_reshaped=copy.deepcopy(probabilities_lossfree_truncated)
        probabilities_lossfree_truncated_reshaped = np.reshape(probabilities_lossfree_full_reshaped,(cutoff ** (nsubspace)))

        probabilities_lossfree_truncated_reshaped = np.reshape(probabilities_lossfree_truncated_reshaped, (cutoff ** (nsubspace)))
        # Computation of the TVD
        tvd_array[i,j] = tvd(probabilities_lossfree_full_reshaped,probabilities_lossfree_truncated_reshaped)
        # if j==len(num_truncatedmodes)-1:
        #     select_threefoldstatistics(probability_tensor_groundthruth=probabilities_lossfree_full_reshaped,
        #                                probability_tensor_experiment=probabilities_lossfree_truncated_reshaped, array_index=array_index,
        #                                numodes=nsubspace, cutoff=cutoff,
        #                                file_title='Histogram_truncation_target_ncoh={:.1f}tarhet_nsqz={:.1f}truncated_modes={:.1f}TVD{:.4f}'.format(
        #                                    target_ncoh, target_nsqz[i], num_truncatedmodes[j], tvd_array[i, j]))
        # Bounds of truncated TVDs
        bound_truncated = qt.tvd_cutoff_bounds(mu=mean_truncated, cov=cov_truncated, cutoff=cutoff)[-1]
        tvd_bound[i,j]= np.sqrt(bound_full ** 2 + bound_truncated ** 2)
        print(i,j)


fig=plt.figure(figsize=plt.figaspect(0.4))
ax=fig.add_subplot(111)
for i in range(len(target_nsqz)):
    ax.plot(num_truncatedmodes,tvd_array[i,:],label=r'$\langle n\rangle=$ {:.2f}'.format(target_nsqz[i]))
    ax.fill_between(num_truncatedmodes,tvd_array[i,:]-tvd_bound[i,:],tvd_array[i,:]+tvd_bound[i,:],alpha=0.2)

ax.yaxis.set_major_locator(LinearLocator(numticks=5))
ax.xaxis.set_major_locator(FixedLocator(num_truncatedmodes))
ax.legend()
ax.set_xlabel('Number of truncated modes')
ax.set_ylabel('TVD')
fig.suptitle(r'Total variation distribution (TVD) of the normalized probability distribution for ncoh={:.1f} and cutoff={:d} as a function of the number of truncated modes and squeezing parameter r'.format(target_ncoh,cutoff),wrap=True)
plt.savefig('Total variation distribution for ncoh={:.1f} for truncated squeezers.pdf'.format(target_ncoh),format='pdf')
plt.savefig('Total variation distribution for ncoh={:.1f} for truncated squeezers.png'.format(target_ncoh),format='png')
plt.tight_layout()
fig.show()
plt.pause(200)
