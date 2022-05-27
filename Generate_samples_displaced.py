import traceback  # For displaying exceptions
import os
import logging
from log_utils import LogUtils
from datetime import datetime  # For current day and time
from datetime import date
from time import time  # For runtime of scripts

from strawberryfields.apps import data, plot, sample, clique
from strawberryfields.apps.sample import postselect
from strawberryfields.decompositions import takagi
from scipy.sparse.csgraph import laplacian
from thewalrus.samples import hafnian_sample_graph
import numpy as np
from numpy.linalg import inv
import networkx as nx
import plotly
import thewalrus.quantum as qt
import thewalrus.samples as sp

def create_directory():
    cwd=os.getcwd()
    today_date=date.today()
    child_name=today_date.strftime("Result\\%d_%m_%Y")
    time_stamp = datetime.now().strftime("%d-%b-%Y-(%H.%M.%S.%f)")
    logging_filename="Results\\{}".format(time_stamp)
    os.makedirs(logging_filename)
    return logging_filename

def create_omega(c,alpha,weight):
    ##Return the rescaling matrix omega, a 2D complex array
    ## c is the scaling coefficient controlling the squeezing required for GBS
    ## alpha is a coefficient that has to be chosen carefully and could introduce a bias in the clique detection
    ## weights is a 1D array of complex coefficients of the diagonal elements of the adjacency matrix of the graph
    n=len(weight)
    omega=c*(np.eye(n,dtype=np.complex64)+alpha*np.diag(weight))
    return omega

def mean_n(squeezing_params,is_lambda=True):
    ##Return the mean photon number for a normal GBS experiment
    ## lambdal is the list of squeezing coefficients
    ## c is the scaling coefficient of the omega matrix
    n = 0
    if is_lambda:
        for i in range(len(squeezing_params)):
            n+=(squeezing_params[i])**2/(1-(squeezing_params[i])**2)
    return n
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
    c=0.1
    weight=np.diag(Adj)
    c1=qt.adj_scaling(Adj,0.45)
    c2=qt.adj_scaling(laplacian(Adj),0.45)
    print(Adj.shape)

    nsamples=100 #number of samples


    #Test between the hafnian_sample_state taking a cov matrix as an argument and hafnian_sample_graph taking an adj matrix and mean photon number
    omega = create_omega(c, 0, weight)


    # With this rescaling convention tanh(ri) can be replaced by tanh(ri)*c**2 where tanh(ri) has been calculated from laplacian(Adj)
    A_rescaled = np.dot(np.dot(omega, laplacian(Adj)), omega)

    (lambdal_rescaled, U_rescaled) = takagi(A_rescaled)

    # Check the mean photon number
    mean_photon_rescaled=mean_n(lambdal_rescaled)
    print(mean_photon_rescaled)


    # Create the covariance matrix of the rescaled adjencency matrix assuming we have a pure gaussian state, i.e the A matrix is diagonal by block with a creation and annihilation operator
    A_matrix=np.block([[A_rescaled,np.zeros((n_subspace,n_subspace))],[np.zeros((n_subspace,n_subspace)),np.conj(A_rescaled)]])
    cov_rescaled=qt.Covmat(inv(np.eye(2*n_subspace) - qt.Xmat(n_subspace)@A_matrix)) #Covariance matrix of the gaussian state, a 2M*2M array
    mu=np.zeros(2*n_subspace) # displacement vector in phase space of dimension 2M

    # Generate the samples
    start=time()
    samples_cov = sp.hafnian_sample_state(cov=cov_rescaled, samples=nsamples)
    hafnian_time = time() - start
    print('Sampling time from the hafnian distribution for N={:.3f}:{:.3f}'.format(nsamples, hafnian_time))


    #Save the samples
    np.savetxt(data_directory + '\\' +'nsamples={:.1f}'.format(nsamples)+'_nsubspace={:.1f}'.format(n_subspace)+'_samples_cov.csv', samples_cov, delimiter=',')



    #Check the number of n-folds detection

    histogram = np.zeros(40)
    for s in samples_cov:
        sum = np.sum(s)
        histogram[int(sum)] += 1
    print('histogram for cov samples',histogram)

    #Postselect and generate the histogram

    samples_cov_2fold=np.array(postselect(samples_cov,2,2))


    samples_cov_2fold=samples_cov_2fold.astype(int)


    histogram_cov_2fold=np.zeros((n_subspace,n_subspace),dtype=np.int32)


    for sample in samples_cov_2fold:
        if np.where(sample==1)[0].size>0:
            indexes=np.where(sample==1)[0]
            index1=indexes[0]
            index2=indexes[1]
            histogram_cov_2fold[index1,index2]+=1
        else:
            index=np.where(sample==2)[0]
            histogram_cov_2fold[index,index]+=1

    hist_cov=[]
    hist_adj=[]
    for i in range(n_subspace):
        for j in range(n_subspace):
            if j>=i:
                hist_cov.append(histogram_cov_2fold[i,j]) #1D array of the 2-fold statistics of GBS for the covariance matrix method

    print(hist_cov)
    print(hist_adj)
    hist_adj=np.array(hist_adj)
    hist_cov=np.array(hist_cov)
    engine_time=time()-start_all
    print('adj normalization',np.sum(hist_adj))
    print('cov normalization', np.sum(hist_cov))
    hist_adj=hist_adj/(np.sum(hist_adj)) #Normalization to convert into probabilities
    hist_cov=hist_cov/(np.sum(hist_cov)) #Normalization to convert into probabilities
    tvd=0.5*np.sum(np.abs(hist_adj-hist_cov))
    print('Total Variation Distance between the covariance matrix and adjacency matrix method for N={:.3f}, {:.3f}'.format(nsamples,tvd))
    print('Sampling time from the hafnian distribution for N={:.3f}'.format(nsamples,hafnian_time))
    print('Total running time{:.3f}'.format(engine_time))
