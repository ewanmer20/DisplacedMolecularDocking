from DGBS_ArbitraryGraph_class import *
from OhSampler import *





# Define the simulation parameters
sim_params = {
    "tau": 1.1,
    "alpha": 2.1,
    "target_nsqz": 2.34,
    "target_ncoh": 10,
    "loss_mode": 0.0,
    "hbar": 2,
    "n_subspace": 24,
    "conv": "real",
    "save": False,
}

# Define the fixed arguments for the postprocessing
fixed_args_postprocessing = {
    "niterations": 8,
}

nsamples=10

# Run the sampler with the optimal parameters to check if the results are consistent
sampler_check=DGBS_Sampler(**sim_params)
result_dic = sampler_check.run_sampler(nsamples=nsamples, foldername="test")
samples = result_dic["samples"]
print("samples:",samples)

#Postprocessing the samples
postprocessing=PostProcessingSamples(samples,sampler_check.Adj)
hist, _, photo_dist = postprocessing.plot_histogram(plot=False, phot_dist=True)
print('uniform sampler',len([list(np.random.choice(len(postprocessing.Adj), np.abs(photo_dist[i]), replace=False)) for i in
                        range(len(postprocessing.cleaned_samples))]))  
samples_oh=[]
for i in range(1,len(hist)):
    if hist[i]>=1:
        g_sample_list=get_G_l_sample(A_in=postprocessing.Adj,N=i,n_samples=int(hist[i]),loss=0.5,fix_photon=True).tolist()
        samples_oh=samples_oh+g_sample_list
    else: 
        pass
print("Oh samples:",samples_oh)

samples_oh_networkx=[np.where(np.array(sample)>=1)[0].tolist() for sample in samples_oh]
print("Oh samples networkx:",len(samples_oh_networkx))
# succ_gbs,succ_uni=postprocessing.plot_success_rate_vs_niter(fixed_args_postprocessing["niterations"],plot=False) 
# succ_gbs,succ_oh=postprocessing.plot_success_rate_vs_OhSampler(fixed_args_postprocessing["niterations"],plot=True)


# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 16))
# ax.plot(np.array(succ_gbs), label='Displaced GBS samples', color='g')
# ax.plot(np.array(succ_uni), label='Uniform samples', color='r')
# ax.plot(np.array(succ_oh), label='Oh samples', color='b')
# ax.set_xlabel('Iteration step of local search algorithm')
# ax.set_ylabel('Success rate (%)')
# ax.grid()
# plt.legend()
# plt.savefig('SuccessRate.png',dpi=300)
# plt.show()




current_dir = os.path.dirname(__file__)
os.chdir(current_dir)




