import traceback  # For displaying exceptions
import os

import logging

import matplotlib.pyplot as plt

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
from thewalrus.symplectic import loss

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

def create_Amatrix(Adj,c,alpha,n_subspace):
    #Create and return the A_matrix used to generate the samples from a GBS experiment
    #Adj is the complete adjacency matrix: not necessarily the one used for the sampling since we can take a submatrix with the dimension tuned by n_subspace!!!!
    #c is the scaling coefficient of the omega matrix
    #alpha is a coefficient that has to be chosen carefully and could introduce a bias in the clique detection
    #nsubpsace is a positive integer for the dimension of the submatrix from the total adjacency matrix to speed-up the sampling
    Adj=Adj[:n_subspace,:n_subspace]
    weight = np.diag(Adj)
    omega = create_omega(c, alpha, weight)
    A_rescaled=np.dot(np.dot(omega, laplacian(Adj)), omega)
    A_matrix = np.block([[A_rescaled, np.zeros((n_subspace, n_subspace))], [np.zeros((n_subspace, n_subspace)), np.conj(A_rescaled)]])
    return A_matrix

def create_cov(Adj,c,alpha,n_subspace,hbar=2):
    #Create a Wigner covariance matrix in the xxpp ordering with hbar=2
    # Adj is the complete adjacency matrix: not necessarily the one used for the sampling!!!!
    # c is the scaling coefficient of the omega matrix
    # alpha is a coefficient that has to be chosen carefully and could introduce a bias in the clique detection
    # nsubpsace is the dimension of the submatrix from the total adjacency matrix to speed-up the sampling
    A_matrix=create_Amatrix(Adj,c,alpha,n_subspace)
    cov_rescaled = qt.Covmat(inv(np.eye(2 * n_subspace) - qt.Xmat(n_subspace) @ A_matrix),hbar=hbar)  # Covariance matrix of the gaussian state, a 2M*2M array
    return cov_rescaled

def samples_cov(Adj,c,alpha,n_subspace,nsamples,data_directory,loss_mode=0,mu=None,hbar=2):
    #Generate samples from the adjacency matrix
    # Adj is the complete adjacency matrix: not necessarily the one used for the sampling!!!!
    # c is the scaling coefficient of the omega matrix
    # alpha is a coefficient that has to be chosen carefully and could introduce a bias in the clique detection
    # nsubpsace is the dimension of the submatrix from the total adjacency matrix to speed-up the sampling
    # nsamples is the number of samples we want to produce
    # data_directory is the directory for the csv results file
    # loss is a float number taking into account the loss
    # mu is the displacement
    # Return a 2D numpy array of samples
    t=1.-loss_mode
    if mu==None:
        mu=np.zeros(2*n_subspace)

    cov_rescaled=create_cov(Adj,c,alpha,n_subspace,hbar=hbar)


    if loss_mode!=0:
        mu_loss=mu.copy()
        cov_loss = cov_rescaled.copy()
        for i in range (n_subspace):
            mu_loss,cov_loss=loss(mu=mu_loss,cov=cov_loss,T=t,nbar=0,mode=i)
        samples = sp.hafnian_sample_state(cov=cov_loss, mean=mu_loss, samples=nsamples,hbar=hbar)
        np.savetxt(data_directory + '\\' + 'nsamples={:.1f}'.format(nsamples) + '_nsubspace={:.1f}'.format(n_subspace) +'loss={:.2f}'.format(loss_mode)+ '_samples_cov.csv', samples, delimiter=',')
    else:
        samples=sp.hafnian_sample_state(cov=cov_rescaled,mean=mu, samples=nsamples)
        np.savetxt(data_directory + '\\' + 'nsamples={:.1f}'.format(nsamples) + '_nsubspace={:.1f}'.format(n_subspace) + '_samples_cov.csv', samples, delimiter=',')
    return samples

def mean_n(squeezing_params,is_lambda=True):
    ##Return the mean photon number for a normal GBS experiment
    ## lambdal is the list of squeezing coefficients
    ## c is the scaling coefficient of the omega matrix
    n = 0
    if is_lambda:
        for i in range(len(squeezing_params)):
            n+=(squeezing_params[i])**2/(1-(squeezing_params[i])**2)
    return n

def hist_coinc(samples,n_subspace):
    #Return normalized histogram from click coincidence between the modes
    samples_2fold = np.array(postselect(samples, 2, 2))
    samples_2fold = samples_2fold.astype(int)
    histogram_2fold = np.zeros((n_subspace, n_subspace), dtype=np.int32)
    for sample in samples_2fold:
        if np.where(sample == 1)[0].size > 0:
            indexes = np.where(sample == 1)[0]
            index1 = indexes[0]
            index2 = indexes[1]
            histogram_2fold[index1, index2] += 1
        else:
            index = np.where(sample == 2)[0]
            histogram_2fold[index, index] += 1
    hist = []
    for i in range(n_subspace):
        for j in range(n_subspace):
            if j >= i:
                hist.append(histogram_2fold[i, j])  # 1D array of the 2-fold statistics of GBS for the covariance matrix method
    return np.array(hist)/np.sum(hist)

def tvd(hist1,hist2):
    #Return the Total Variation distance between two normalized distribution hist1 and hist2, two 1D numpy arrays of same size
    return 0.5*np.sum(np.abs(hist1-hist2))










