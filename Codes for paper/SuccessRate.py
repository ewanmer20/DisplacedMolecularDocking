from DGBS_ArbitraryGraph_class import *

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
    "niterations": 7,
}

nsamples=5000

# Run the sampler with the optimal parameters to check if the results are consistent
sampler_check=DGBS_Sampler(**sim_params)
result_dic = sampler_check.run_sampler(nsamples=nsamples, foldername="test")
samples = result_dic["samples"]
print(" first samples:",samples[:20])

#Postprocessing the samples
postprocessing=PostProcessingSamples(samples,sampler_check.Adj)
succ_gbs,succ_uni=postprocessing.plot_success_rate_vs_niter(fixed_args_postprocessing["niterations"],plot=False) 
succ_gbs,succ_oh=postprocessing.plot_success_rate_vs_OhSampler(fixed_args_postprocessing["niterations"],plot=False)

current_dir = os.path.dirname(__file__)
os.chdir(current_dir)
results={"succ_gbs":succ_gbs,"succ_uni":succ_uni,"succ_oh":succ_oh,"samples":samples.flatten(),"sim_params":sim_params}
pd_result=pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))
pd_result.to_csv("raw_data_TaceAs25112024_1000_samples.csv")
plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 16))
ax.plot(np.array(succ_gbs), label='Displaced GBS samples', color='g')
ax.plot(np.array(succ_uni), label='Uniform samples', color='r')
ax.plot(np.array(succ_oh), label='Oh samples', color='b')
ax.set_xlabel('Iteration step of local search algorithm')
ax.set_ylabel('Success rate (%)')
ax.grid()
plt.legend()
# plt.savefig('SuccessRate.png',dpi=300)
plt.savefig('SuccessRate10000.svg')
plt.show()









