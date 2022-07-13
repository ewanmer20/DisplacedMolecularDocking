import traceback  # For displaying exceptions
import os
import csv

from log_utils import LogUtils
from datetime import datetime  # For current day and time
from datetime import date
from time import time  # For runtime of scripts

from strawberryfields.apps import data, plot, sample, clique
from strawberryfields.apps.sample import postselect
from strawberryfields.decompositions import takagi
from scipy.sparse.csgraph import laplacian
from scipy.optimize import minimize_scalar
from thewalrus.samples import hafnian_sample_graph
import numpy as np
from numpy.linalg import inv
import thewalrus.quantum as qt
import thewalrus.samples as sp
from thewalrus.symplectic import loss
EXP_PATH=os.getcwd()

def create_directory():
    cwd=os.getcwd()
    today_date=date.today()
    child_name=today_date.strftime("Result\\%d_%m_%Y")
    time_stamp = datetime.now().strftime("%d-%b-%Y-(%H.%M.%S.%f)")
    logging_filename="Results\\{}".format(time_stamp)
    os.makedirs(logging_filename)
    return logging_filename

def get_my_keys(tau):

    # function to retrieve the keys for the BIG matrix, according to the
    # formatting produced by make_adj.py
    # # tau is the flexibility constant used to define the adjacency matrix
    # returns a list of list of strings

    raw_keys = []
    with open(EXP_PATH + "/big/key_tau" + str(tau) + "_.csv",
              newline="") as csvfile:
        keymaker = csv.reader(csvfile, delimiter=";", quotechar="'")
        for row in keymaker:
            raw_keys.append(row)
    raw_keys = raw_keys[0]
    raw_keys.pop()

    # put the keys in list of list format
    list_keys = []
    for key in raw_keys:
        p_type = key[1:4]
        l_type = key[5:-1]
        while p_type[-1].isdigit():
            p_type = p_type[:-1]
        while l_type[-1].isdigit():
            l_type = l_type[:-1]

        list_keys.append([p_type, l_type])

    return list_keys

def make_potential_vect():
    """
    function to generate the potential matrix given the potential value in Banchi et al. using the same formatting
    """
    ligand_dists, pocket_dists, ligand_key, pocket_key = get_data()

    v_set = [[i, j] for i in range(len(ligand_key))
             for j in range(len(pocket_key))]

    potential_vect=[]
    potential_data=np.array([[0.5244,0.6686,0.1453],[0.6686,0.5478,0.2317],[0.1453,0.2317,0.0504]])
    # potential data given in the table S1 given in Banchi et al.: First coloumn: Hydrogen-bond donor (HD)
    # Second column: Hydrogen-bond acceptor (HA) and in the last column: Hydrophobe (Hp). To change the ordering, one have to change the mapping function and the potential matrix
    for vertex in v_set:
        row=str(ligand_key[vertex[0]])[0:2]
        column=str(pocket_key[vertex[1]])[0:2]
        row_index=mapping(row)
        column_index=mapping(column)
        potential_vect.append(potential_data[row_index,column_index])
    potential_vect=np.array(potential_vect)
    return potential_vect

def get_data():

    ligand_dists = np.array([
        [0.0, 4.6, 9.1, 9.9],
        [0.0, 0.0, 8.1, 8.4],
        [0.0, 0.0, 0.0, 1.2],
        [0.0, 0.0, 0.0, 0.0],
    ])
    ligand_dists = ligand_dists + ligand_dists.T
    ligand_key = ["HD1", "HA1", "Hp1", "Hp2"]


    pocket_dists = np.array([
        [0.0, 2.8, 4.6, 7.6, 5.9, 11.1],
        [0.0, 0.0, 2.7, 5.1, 3.6, 10.5],
        [0.0, 0.0, 0.0, 3.9, 3.5, 12.0],
        [0.0, 0.0, 0.0, 0.0, 2.2, 10.6],
        [0.0, 0.0, 0.0, 0.0, 0.0, 9.00],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.00],

    ])
    pocket_dists = pocket_dists + pocket_dists.T
    pocket_key = ["HD1", "HD2", "HA1", "HA2", "HA3", "Hp1"]


    return ligand_dists, pocket_dists, ligand_key, pocket_key


def mapping(pharmacopore):
            if pharmacopore=='HD':
                return 0
            elif pharmacopore=='HA':
                return 1
            elif pharmacopore=='Hp':
                return 2




def make_omega(renorm, alpha):
    """
    function to generate the rescaling matrix omega, as defined in Banchi et.
    al.
    renorm is a positive scalar that is supposed to control the amount squeezing required
    alpha is the strength of the weigth potentials in the matrix

    returns a 2-d numpy array
    """
    # generate vertex weights

    # big_potentials = []
    # print(list_keys)
    # for pair in list_keys:
    #     row = list_keys.index(pair[0])
    #     col = list_keys.index(pair[1])
    #     big_potentials.append(potential_mat[row, col])
    big_potentials=make_potential_vect()


    # generate the rescaling matrix Omega
    # c and alpha are tunable parameters
    # WARNING: they must be carefully chosen.
    omega = renorm * (np.eye(len(big_potentials)) +
                      alpha * np.diag(big_potentials))
    return omega


# define pharmacophore interaction potentials in a matrix
# these are copied straight from table S1 in Banchi et al.



def create_Amatrix(Adj,c,alpha,n_subspace,tau=1.1):
    #Create and return the A_matrix used to generate the samples from a GBS experiment
    #Adj is the complete adjacency matrix: not necessarily the one used for the sampling since we can take a submatrix with the dimension tuned by n_subspace!!!!
    #c is the scaling coefficient of the omega matrix
    #alpha is a coefficient that has to be chosen carefully and could introduce a bias in the clique detection
    #nsubpsace is a positive integer for the dimension of the submatrix from the total adjacency matrix to speed-up the sampling
    #tau is the flexibility constant used to define the adjacency matrix with the formatting from make_adj.py. The default value for tau is the one used for Tace-As in Banchi et al.
    Adj=Adj[:n_subspace,:n_subspace]
    omega = make_omega(c, alpha)[:n_subspace,:n_subspace]
    BIG=np.dot(np.dot(omega, laplacian(Adj)), omega)
    A_matrix = np.block([[BIG, np.zeros((n_subspace, n_subspace))], [np.zeros((n_subspace, n_subspace)), np.conj(BIG)]])
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
    # loss is a float number taking into account total loss of the GBS experiment including: coupling and collection efficiency, transmission, fiber coupling and detection efficiency
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
        np.savetxt(data_directory + '\\' + 'nsamples={:.1f}'.format(nsamples)+ '_nsubspace={:.1f}'.format(n_subspace) +'loss={:.2f}'.format(loss_mode)+ '_samples_cov.csv', samples, delimiter=',')
    else:
        samples=sp.hafnian_sample_state(cov=cov_rescaled,mean=mu, samples=nsamples)
        np.savetxt(data_directory + '\\' + 'nsamples={:.1f}'.format(nsamples) + '_nsubspace={:.1f}'.format(n_subspace) + '_samples_cov.csv', samples, delimiter=',')
    return samples

def mean_n(BIG):
    ##Return the mean photon number for a normal GBS experiment
    ## BIG is the binding interaction graph, a numpy array
    n = 0
    (lambdal_rescaled, U_rescaled) = takagi(BIG)
    for i in range(len(lambdal_rescaled)):
        n+=(lambdal_rescaled[i])**2/(1-(lambdal_rescaled[i])**2)
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

def tune_c(alpha,target_n,Adjtot,nsubpsace):
    #Return the c parameter such that the GBS experiment reaches a mean photon number target
    #alpha=the alpha at the input of the adjacency matrix
    #target_n=a positive number representing  target mean photon n
    #Adjtot= the adjacency matrix of the total graph
    #nsubspace= the dimension of the considered subspace
    Adj = Adjtot[:nsubpsace, :nsubpsace]
    def cost(c,alpha,target_n,Adj,n_subspace):
        omega = make_omega(c, alpha)[:n_subspace, :n_subspace]
        BIG = np.dot(np.dot(omega, laplacian(Adj)), omega)
        return np.abs(target_n-mean_n(BIG))
    res=minimize_scalar(cost,args=(alpha,target_n,Adj,nsubpsace))
    return res.x









