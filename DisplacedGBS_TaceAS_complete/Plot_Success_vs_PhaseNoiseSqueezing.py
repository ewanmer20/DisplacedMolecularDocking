
from Library_Generate_displaced_samples_alternative_encoding_TaceAs import *
from Library_Analysis import*
from matplotlib.ticker import IndexLocator,LinearLocator

def optimize_displacement(target_ncoh,Sigma_Q,omega,weights,nsubspace,hbar=2):
    """
     :param target_ncoh: the target in terms of mean photon number from displacement
    :param omega: the rescaling matrix
    :param weights: the weights of the nodes
    :return: optimize the parameters of kappa and delta to get mean photon from displacement as close as possible to the target
    """

    def cost(params,target_ncoh,Sigma_Q,weights,nsubspace,hbar=2):
        gamma=give_gamma(params[0],params[1],omega,weights,nsubspace)
        d_alpha = (Sigma_Q @ gamma)[:nsubspace]
        mean_rescaled=np.sqrt(2*hbar)*np.concatenate([d_alpha, np.zeros(nsubspace)])
        ncoh = np.sum(np.abs(mean_rescaled) ** 2) / (2 * hbar)
        return (ncoh-target_ncoh)**2
    res=minimize(cost,args=(target_ncoh,Sigma_Q,weights,nsubspace,hbar),bounds=Bounds([0.1,1.],[np.inf,np.inf]),x0=[1.,1.])
    return res


def create_cov_mean_alt_phasenoise(Adj,c,alpha,target_ncoh,nsubspace,hbar=2,tau=1.1,conv='real',phase_noise=0.1):
    '''
    Create and return the covriance matrix and mean vector used to generate the samples from a GBS experiment assuming uniform phase noisy on the input squeezed states
    :param Adj:is the complete adjacency matrix: not necessarily the one used for the sampling since we can take a submatrix with the dimension tuned by n_subspace!!!!
    :param c: is a rescaling coefficient
    :param alpha: is a coefficient that has to be chosen carefully and could introduce a bias in the clique detection
    :param target_ncoh: target mean photon number for displacement that needs to be optimized independently from the squeezing
    :param n_subspace: a positive integer for the dimension of the submatrix from the total adjacency matrix to speed-up the sampling
    :param tau: is the flexibility constant used to define the adjacency matrix with the formatting from make_adj.py. The default value for tau is the one used for Tace-As in Banchi et al.
    :param conv: if complex return the outputs in the complex convention in the ordering aadag, else if real returns in the xxpp real convention(the one used by the Walrus!!)
    :param phase_noise: is the standard deviation of the phase noise added to the tanh of the eigenvalues of the BIG matrix
    :return:
    '''
    Id = np.eye(nsubspace)
    weights=make_potential_vect()[:nsubspace]
    Adj=Adj[:nsubspace,:nsubspace]
    omega = make_omega(c, alpha)[:nsubspace,:nsubspace]
    BIG=omega@Adj@omega
    (tanhr_array, U) = takagi(BIG)
    tanhr_array_noisy=[tanh*np.exp(1j*np.random.normal(loc=0,scale=phase_noise)*np.pi)for tanh in tanhr_array]
    BIG_noisy = U @ np.diag(tanhr_array_noisy) @ U.T
    Sigma_Qinv = np.block([[Id, -np.conj(BIG_noisy)], [-BIG_noisy, Id]])
    Sigma_Q = inv(Sigma_Qinv)
    params=optimize_displacement(Sigma_Q=Sigma_Q,target_ncoh=target_ncoh,omega=omega,weights=weights,nsubspace=nsubspace,hbar=hbar).x
    gamma=give_gamma(kappa=params[0],delta=params[1],omega=omega,weights=weights,nsubspace=nsubspace)
    # gamma = np.block([[omega, np.zeros((nsubspace,nsubspace))], [np.zeros((nsubspace,nsubspace)), omega]])@ np.concatenate(((1+1.1*weights)**2,(1+1.1*weights)**2))
    d_alpha=(Sigma_Q @ gamma)[:nsubspace]
    if conv=='real':
        return qt.Covmat(Sigma_Q,hbar=hbar),np.sqrt(2*hbar)*np.concatenate([[d.real for d in d_alpha],[d.imag for d in d_alpha]])


    elif conv=='complex':
        return Sigma_Q-np.eye(2 * nsubspace) / 2,np.sqrt(2*hbar)*np.concatenate([d_alpha,np.conj(d_alpha)])


def samples_cov_alt_phasenoise(Adj,alpha,target_nsqz,target_ncoh,n_subspace,nsamples,data_directory,loss_mode=0,hbar=2,phase_noise=0.1):
    '''
    Generate samples from the adjacency matrix with the encoding based on BIG=c(1+alpha*weigths)*Adj*c(1+alpha*weigths)

    :param Adj: the complete adjacency matrix of the graph
    :param nsqz_target:  is the target for the mean photon number coming from the squeezing
    :param taarget_ncoh: target mean photon number for displacement that needs to be optimized independently from the squeezing
    :param alpha: is a coefficient that has to be chosen carefully and could introduce a bias in the clique detection
    :param n_subspace: a positive integer for the dimension of the submatrix from the total adjacency matrix to speed-up the sampling
    :param nsamples: the number of samples we want to produce
    :param data_directory:
    :param loss_mode: a float number taking into account total loss of the GBS experiment including: coupling and collection efficiency, transmission, fiber coupling and detection efficiency at each mode at the end of the interferometer
    :param hbar:
    :param phase_noise: is the standard deviation of the phase noise added to the tanh of the eigenvalues of the BIG matrix
    :return: Return a 2D numpy array of samples
    '''
    t=1.-loss_mode
    c=tune_c(alpha,target_nsqz,Adj,n_subspace)
    omega = make_omega(c, alpha)[:n_subspace, :n_subspace]
    BIG = np.dot(np.dot(omega, laplacian(Adj)), omega)
    print("Mean photon number from squeezing:",mean_nsqz(BIG))
    cov_rescaled,mean_rescaled=create_cov_mean_alt_phasenoise(Adj,c,alpha,target_ncoh,n_subspace,hbar=hbar,phase_noise=phase_noise) #covariance matrix and mean matrix given the parameters c and v, alpha and Adj
    ncoh=np.sum(np.abs(mean_rescaled)**2)/(2*hbar)# Mean photon number with a mean vector in the xxpp ordering

    print("Mean photon number from displacement:",ncoh)
    path = data_directory + '\\' + 'nsamples={:.1f}'.format(nsamples) + '_nsubspace={:.1f}'.format(n_subspace) + 'alpha={:.1f}'.format(alpha) + 'loss={:.2f}'.format(loss_mode) + 'ncoh={:2f}'.format(ncoh) + '_displaced_samples_cov.csv'

    if loss_mode!=0:
        mu_loss=mean_rescaled.copy()
        cov_loss = cov_rescaled.copy()
        for i in range (n_subspace):
            mu_loss,cov_loss=loss(mu=mu_loss,cov=cov_loss,T=t,nbar=0,mode=i)
        samples = sp.torontonian_sample_state(cov=cov_loss, mu=mu_loss, samples=nsamples,hbar=hbar)

        np.savetxt(path, samples, delimiter=',')
    else:

        samples=sp.torontonian_sample_state(cov=cov_rescaled,mu=mean_rescaled, samples=nsamples)
        np.savetxt(path, samples, delimiter=',')
    return samples,path,ncoh,mean_nsqz(BIG)

cwd='big\\big_tau1.1_.csv'
BIG=log_data(cwd)

plt.close('all')
nsubspace=24
tau=1.1
alpha=2.1
plt.rcParams.update({'font.size': 28})
start_all=time()
Adj,_=make_adj(tau)

# np.savetxt('Adjacency_matrix _tau1.1_test.csv',Adj,delimiter=',',fmt='%d')
nsamples=20000

target_nsqz=2.34
target_ncoh=10
n_iterations_local_search=7
loss_mode=0.01


data_directory=create_directory()
n_trials=10
# target_ncoh=np.linspace(3,10,5)
# target_nsqz=np.linspace(0.1,2,15)
# succ_sqzcoh_gbs=np.zeros((len(target_ncoh),len(target_nsqz)))
# succ_sqzcoh_uni=np.zeros((len(target_ncoh),len(target_nsqz)))
phase_noise_array=np.linspace(0,1,10)
succ_result_array=[]
counter=0
for phase_noise in phase_noise_array:
    array_trial=[]
    print("Element number in phase_noise_array:",counter)
    for i in range(n_trials):

        samples, path, ncoh, nsqz= samples_cov_alt_phasenoise(Adj=Adj, alpha=alpha, target_nsqz=target_nsqz, target_ncoh=target_ncoh,nsamples=nsamples, data_directory=data_directory, loss_mode=loss_mode,hbar=2, n_subspace=nsubspace,phase_noise=phase_noise)
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
        array_trial.append(succ_gbs[-1])
        print("Trial number:",i)
        
    succ_result_array.append(np.mean(array_trial))
    counter+=1

    
    

np.savetxt('phase_noise_succ_gbs_ncoh={:.1f}_nsqz={:.1f}.txt'.format(ncoh,nsqz), np.array(succ_result_array), delimiter=',')
np.savetxt('phase_noise_array_ncoh={:.1f}_nsqz={:.1f}.txt'.format(ncoh,nsqz), np.array(phase_noise_array), delimiter=',')

fig=plt.figure(figsize=plt.figaspect(0.4))
ax=fig.add_subplot(121)

ax.plot(phase_noise_array,succ_result_array)   
ax.yaxis.set_major_locator(LinearLocator(numticks=5))
ax.xaxis.set_major_locator(IndexLocator(base=0.1,offset=0))

ax.set_xlabel('Noise standard deviation, '+r'$\sigma/\pi$')
ax.set_ylabel('Success rate')
plt.show()


