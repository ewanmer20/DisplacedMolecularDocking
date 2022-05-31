from Generate_samples import*


if __name__=='__main__':
    #Convention hbar=2 like the one used in Strawberry fields and The Walrus
    # The convention used is an xxpp ordering and handling position, momentum operatorss
    LogUtils.log_config('Generate_samples')
    start_all=time()
    n_subspace=10 # Has to be less or equal to 24
    data_directory = create_directory()
    TA = data.TaceAs()
    Adj = TA.adj
    Adj=Adj[:n_subspace,:n_subspace]
    alpha=0
    c=0.2
    weight=np.diag(Adj)
    c1=qt.adj_scaling(Adj,0.45)
    c2=qt.adj_scaling(laplacian(Adj),0.45)
    print(Adj.shape)
    nsamples=10000 #number of samples
    alpha=np.logspace(start=0.10,stop=2.0,num=6)
    tvd_hist=[]
    # Test between the hafnian_sample_state taking a cov matrix as an argument and hafnian_sample_graph taking an adj matrix and mean photon number
    omega = create_omega(c, 0, weight)

    # With this rescaling convention tanh(ri) can be replaced by tanh(ri)*c**2 where tanh(ri) has been calculated from laplacian(Adj)
    A_rescaled = np.dot(np.dot(omega, laplacian(Adj)), omega)

    (lambdal, U) = takagi(laplacian(Adj))
    (lambdal_rescaled, U_rescaled) = takagi(A_rescaled)

    # Check the mean photon number
    mean_photon_rescaled = mean_n(lambdal_rescaled)
    print(mean_photon_rescaled)

    samples_adj = hafnian_sample_graph(laplacian(Adj), mean_photon_rescaled, samples=nsamples)
    hist_adj = hist_coinc(samples_adj, n_subspace)
    for i in range (len(alpha)):
        samples=samples_cov(Adj,c,alpha[i],n_subspace,nsamples,data_directory,hbar=2)
        hist_cov=hist_coinc(samples,n_subspace)
        tvd_hist.append(tvd(hist_adj,hist_cov))
        print(i)

    time=time()-start_all
    tvd_hist=np.array(tvd_hist)
    np.savetxt(data_directory + 'tvd_hist_alpha.txt', tvd_hist, delimiter=',')
    np.savetxt(data_directory + 'alpha_array.txt', alpha, delimiter=',')
    print('Total running time{:.3f}'.format(time))
    plt.figure(figsize=(16,16))
    plt.plot(alpha,tvd_hist)
    plt.xscale("log")
    plt.xlabel("Alpha value in log scale")
    plt.ylabel("two-fold Total Variation Distance compared to alpha=0")
    plt.savefig(data_directory+'plot_tvd_vs_alpha.png')
    plt.show()
