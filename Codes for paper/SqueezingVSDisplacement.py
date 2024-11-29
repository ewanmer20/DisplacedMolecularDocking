from DGBS_ArbitraryGraph_class import *
from matplotlib import cm
# Define the simulation parameters
sim_params = {
    "tau": 1.1,
    "alpha": 2.1,
    "target_nsqz": 2.34,
    "target_ncoh": 10,
    "loss_mode": 0.5,
    "hbar": 2,
    "n_subspace": 24,
    "conv": "real",
    "save": True,
}

# Define the fixed arguments for the postprocessing
fixed_args_postprocessing = {
    "niterations": 7,
}



ncoh_min=3
ncoh_max=15
nsqz_min=1
nsqz_max=4
nsamples=1000
foldername_data="data_success_rate_vs_disp_vs_sqz_2"
n_points=10




x_target_ncoh=np.linspace(ncoh_min,ncoh_max,n_points)
y_target_nsqz=np.linspace(nsqz_min,nsqz_max,n_points)
succ_sqzcoh_gbs=np.zeros((n_points,n_points))
succ_sqzcoh_uni=np.zeros((n_points,n_points))
# Run the sampler with the optimal parameters to check if the results are consistent
sampler=DGBS_Sampler(**sim_params)
for i in range(n_points):
    for j in range(n_points):
        start_all=time()
        result_dic=sampler.run_sampler(nsamples=nsamples, foldername=foldername_data,data_id=f"{i}_{j}")
        sampler.target_nsqz=y_target_nsqz[j]
        sampler.target_ncoh=x_target_ncoh[i]
        samples = result_dic["samples"]
        print("First samples:",samples[:5])
        #Postprocessing the samples
        postprocessing=PostProcessingSamples(samples,sampler.Adj)
        succ_gbs,succ_uni=postprocessing.plot_success_rate_vs_niter(fixed_args_postprocessing["niterations"],plot=False)
        succ_sqzcoh_gbs[i,j]=succ_gbs[-1]
        succ_sqzcoh_uni[i,j] =succ_uni[-1]
        time1 = time() - start_all
        print(f"Time for iteration i={i} and j={j} is {time1}")



os.chdir(sampler.data_directory)
results={"succ_gbs":succ_sqzcoh_gbs.flatten(),"succ_uni":succ_sqzcoh_uni.flatten(),"sim_params":sim_params}
pd_result=pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))
pd_result.to_csv("matrices_TaceAs25112024.csv")


plt.rcParams['font.size']=12
fig=plt.figure(figsize=plt.figaspect(0.4))
ax=fig.add_subplot(1,2,1)
array_gbs=ax.imshow(succ_sqzcoh_gbs,cmap=cm.viridis,extent=[ncoh_min,ncoh_max,nsqz_max,nsqz_min])
fig.colorbar(array_gbs,shrink=0.5)
ax.set_title('Success rate vs input mean photon number of sqz\n and dispwith a DGBS sampler \nafter {:.1f} iter and {:.1f} loss'.format(fixed_args_postprocessing["niterations"],sim_params["loss_mode"]) )
ax.set_xlabel(r'$\langle n_{coh}\rangle$')
ax.set_ylabel(r'$\langle n_{sqz}\rangle$')


ax=fig.add_subplot(1,2,2)
array_uni=ax.imshow(succ_sqzcoh_uni,cmap=cm.viridis,extent=[ncoh_min,ncoh_max,nsqz_max,nsqz_min])
fig.colorbar(array_uni,shrink=0.5)
ax.set_title('Success rate vs input mean photon number of sqz\n and disp with a uniform sampler after \n{:.1f} iter and {:.1f}  loss'.format(fixed_args_postprocessing["niterations"],sim_params["loss_mode"]) )
ax.set_xlabel(r'$\langle n_{coh}\rangle$')
ax.set_ylabel(r'$\langle n_{sqz}\rangle$')
plt.tight_layout()
plt.savefig('Success Rate vs Disp and Sqz TaceAs,nsamples={:.1f},nsqz_min={:.1f},nsqz_max={:.1f},ncoh_min={:.1f},ncoh_max={:.1f},loss={:.1f},niteration={:.1f}.pdf'.format(nsamples,nsqz_min,nsqz_max,ncoh_min,ncoh_max,sim_params["loss_mode"],fixed_args_postprocessing["niterations"]),format='pdf')
fig.show()
plt.pause(10000)