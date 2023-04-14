import csv

import matplotlib.pyplot as plt
import numpy as np

from Generate_samples import*
from Analysis_lib import*
plt.rcParams.update({'font.size': 28})
if __name__=='__main__':
    # #Convention hbar=2 like the one used in Strawberry fields and The Walrus
    # # The convention used is an xxpp ordering and handling position, momentum operatorss
    LogUtils.log_config('Generate_samples')
    start_all=time()
    n_subspace=24 # Has to be less or equal to 24
    n_phot_target=2.34
    TA = data.TaceAs()
    Adj = TA.adj
    alpha=2.1
    tau=1.1 #Default value for the flexibility constant in Tace-As
    print(Adj.shape)

    data_directory = create_directory()
    print(data_directory)
    c=tune_c(alpha=alpha,target_n=n_phot_target,Adjtot=Adj,nsubpsace=24)
    omega = make_omega(c,alpha)
    BIG = np.dot(np.dot(omega, laplacian(Adj)), omega)
    print(mean_n(BIG))
    start_all=time()

    samples = samples_cov(Adj, c, alpha, n_subspace=24, nsamples=10000, data_directory=data_directory,loss_mode=0.5, hbar=2)
    time1 = time() - start_all
    print(time1)
    succ_GBS,succ_uni=plot_success_rate_vs_niter(samples,Adj,niter=7,weights=make_potential_vect())
    # nsamples=100000 #number of samples
    # nsamples_list=np.logspace(2,5,20)
    # tvd_hist=[]
    # omega = make_omega(c,0)[:10,:10]
    # BIG_reduced= np.dot(np.dot(omega, laplacian(Adj_reduced)), omega)
    # mean_photon_rescaled=mean_n(BIG_reduced)
    # # samples_adj = hafnian_sample_graph(BIG_reduced, mean_photon_rescaled, samples=nsamples)
    # samples_adj=[]
    # with open('Reference_samples/Reference_mean_n=0.89 nsamples=100000.0_nsubspace=10.0_samples_cov.csv') as reference_data:
    #     csv_reader=csv.reader(reference_data,delimiter=',')
    #     for row in csv_reader:
    #         samples_adj.append(row)
    # samples_adj=np.array(samples_adj)
    # samples_adj=samples_adj.astype(np.float64)
    # print(samples_adj[:5,:])
    #
    # hist_adj = hist_coinc(samples_adj, n_subspace)
    # print(mean_photon_rescaled)
    # # samples=samples_cov(Adj,c,0,n_subspace,nsamples,data_directory,hbar=2)
    # # hist_cov = hist_coinc(samples, n_subspace)
    # # tvd_hist.append(tvd(hist_adj,hist_cov))
    # # print(tvd_hist)
    #
    #
    #
    #
    #
    # for i in range(len(nsamples_list)):
    #     # Test between the hafnian_sample_state taking a cov matrix as an argument and hafnian_sample_graph taking an adj matrix and mean photon number
    #
    #     # With this rescaling convention tanh(ri) can be replaced by tanh(ri)*c**2 where tanh(ri) has been calculated from laplacian(Adj)
    #     omega = make_omega(c,alpha)[:10, :10]
    #     BIG_reduced = np.dot(np.dot(omega, laplacian(Adj_reduced)), omega)
    #     samples=samples_cov(Adj,c,alpha,n_subspace,nsamples_list[i],data_directory,hbar=2)
    #     hist_cov=hist_coinc(samples,n_subspace)
    #     tvd_hist.append(tvd(hist_adj,hist_cov))
    #
    #     # Check the mean photon number
    #
    #     print(mean_n(BIG_reduced))
    #     print(i)
    #
    #
    # time=time()-start_all
    # tvd_hist=np.array(tvd_hist)
    # np.savetxt(data_directory + 'tvd_hist_nsamples.txt', tvd_hist, delimiter=',')
    # np.savetxt(data_directory + 'nsamples_array.txt', nsamples_list, delimiter=',')
    # print('Total running time{:.3f}'.format(time))
    # # tvd_hist=np.loadtxt('Results/31-May-2022-(15.02.40.187091)tvd_hist_loss.txt')
    # plt.figure(figsize=(16,16))
    # plt.plot(nsamples_list,tvd_hist)
    # plt.xscale('log')
    # plt.xlabel("Number of samples")
    # plt.ylabel("two-fold Total Variation Distance as a function of number of samples")
    # plt.savefig(data_directory+'plot_tvd_vs_nsamples.png')
    # plt.show()
