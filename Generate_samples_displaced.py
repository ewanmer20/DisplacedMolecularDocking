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
from Generate_samples import *
EXP_PATH=os.getcwd()
def create_directory():
    cwd=os.getcwd()
    today_date=date.today()
    child_name=today_date.strftime("Result_Displacement\\%d_%m_%Y")
    time_stamp = datetime.now().strftime("%d-%b-%Y-(%H.%M.%S.%f)")
    logging_filename="Results\\{}".format(time_stamp)
    os.makedirs(logging_filename)
    return logging_filename

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
nsamples=5000 #number of samples
samples=samples_cov(Adj,c,alpha,n_subspace,nsamples,data_directory,hbar=2)
hist_cov=hist_coinc(samples,n_subspace)
#Test between the hafnian_sample_state taking a cov matrix as an argument and hafnian_sample_graph taking an adj matrix and mean photon number
omega = create_omega(c, 0, weight)


# With this rescaling convention tanh(ri) can be replaced by tanh(ri)*c**2 where tanh(ri) has been calculated from laplacian(Adj)
A_rescaled = np.dot(np.dot(omega, laplacian(Adj)), omega)

(lambdal, U) = takagi(laplacian(Adj))
(lambdal_rescaled, U_rescaled) = takagi(A_rescaled)

# Check the mean photon number
mean_photon_rescaled=mean_n(lambdal_rescaled)
print(mean_photon_rescaled)

samples_adj = hafnian_sample_graph(laplacian(Adj), mean_photon_rescaled, samples=nsamples)
hist_adj=hist_coinc(samples_adj,n_subspace)

np.savetxt(data_directory + '\\' +'nsamples={:.1f}'.format(nsamples)+'_nsubspace={:.1f}'.format(n_subspace)+'_samples_adj.csv', samples_adj, delimiter=',')
tvd_v=tvd(hist_adj,hist_cov)

time=time()-start_all
print('Total Variation Distance between the covariance matrix and adjacency matrix method for N={:.3f}, {:.3f}'.format(nsamples,tvd_v))
# print('Sampling time from the hafnian distribution for N={:.3f}'.format(nsamples,hafnian_time))
print('Total running time{:.3f}'.format(time))



