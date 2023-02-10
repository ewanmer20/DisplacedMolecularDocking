from Generate_displaced_samples_alternative_encoding import *
from Analysis_lib import*
from time import time
cwd='big\\big_tau1.1_.csv'
BIG=log_data(cwd)
nsubspace=24
Adj = data.TaceAs().adj[:nsubspace,:nsubspace]
alpha=2.1
target_ncoh=1
start_all=time()
nsamples=10000

loss_mode=0.5
data_directory = create_directory()

target_nsqz=3
samples,ncoh,path = samples_cov_alt(Adj=Adj,alpha=alpha,target_nsqz=target_nsqz,nsamples=nsamples,data_directory=data_directory,loss_mode=loss_mode,hbar=2,n_subspace=nsubspace)
time1 = time() - start_all
print("Time to create the samples in seconds:",time1)
#Using the adjacency matrix reduced to the first 10 modes to be consistent with the GBS simulation
Adj =data.TaceAs().adj

#Retrieving the potential values for the adjacency matrix
weights=make_potential_vect()

# Histogram of the number of photons per sample
a,nmax=plot_histogram(samples,plot=True)
print(a)

# #Remove the non-zero clicks event and the collision events (just for PNRS detectors)
cleaned_samples,ncollision=clean_samples(samples,nmax)
print(len(cleaned_samples))
print("Number of collisions:",ncollision)

#Find the maximum clique with classical algorithm

clique_max,clique_weight=find_max_clique(BIG,weights,networkx_conv=False)
print('Max clique',clique_max)
print('Weights of the max clique',clique_weight)

searched_gbs,searched_uni=plot_success_rate_vs_niter(cleaned_GBS_samples=cleaned_samples,Adj=BIG,weights=weights,niter=7)
plot_histogram_clique_values(cleaned_GBS_samples=cleaned_samples,nmax=nmax,Adj=Adj,weights=weights,plot=True)

searched_gbs_det=networkx_distribution_to_clicks(searched_gbs,nsubspace)
searched_uni_det=networkx_distribution_to_clicks(searched_uni,nsubspace)

# a,b=plot_histogram(searched_gbs_det,plot=False)
# a1,b1=plot_histogram(searched_uni_det,plot=False)
# plot_histogram_clique_values(cleaned_GBS_samples=searched_gbs_det,nmax=b,Adj=Adj,weights=weights,plot=True)
# plot_histogram_clique_values(cleaned_GBS_samples=searched_uni_det,nmax=b1,Adj=Adj,weights=weights,plot=True)
#
