import traceback  # For displaying exceptions
import os
import logging
from log_utils import LogUtils
import numpy as np
from datetime import datetime  # For current day and time
from datetime import date
import csv
from scipy.linalg import eigvalsh,inv
from scipy.optimize import minimize
import thewalrus.random as rd
import thewalrus.quantum as qt
import thewalrus.samples as sp
from thewalrus.symplectic import loss
from strawberryfields.apps import data
from strawberryfields.decompositions import takagi

from scipy.sparse.csgraph import laplacian
from scipy.optimize import minimize_scalar
EXP_PATH=os.getcwd()
def create_directory():
    cwd=os.getcwd()
    today_date=date.today()
    child_name=today_date.strftime("Results_alt_encoding\\%d_%m_%Y")
    time_stamp = datetime.now().strftime("%d-%b-%Y-(%H.%M.%S.%f)")
    logging_filename="Results_alt_encoding\\{}".format(time_stamp)
    os.makedirs(logging_filename)
    return logging_filename

def log_data(csv_file):
    #Return from the csv file the samples in a numpy array
    tot_samples=[]
    with open(csv_file) as reference_data:
        csv_reader = csv.reader(reference_data, delimiter=',')
        for row in csv_reader:
            tot_samples.append(row)
    tot_samples = np.array(tot_samples)
    tot_samples = tot_samples.astype(np.float64)

    return tot_samples


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

    big_potentials=make_potential_vect()
    # generate the rescaling matrix Omega
    # c and alpha are tunable parameters
    # WARNING: they must be carefully chosen.
    omega = renorm * (np.eye(len(big_potentials)) +
                      alpha * np.diag(big_potentials))
    return omega


def make_omega(c,alpha):
    """""
    function to generate a more generalized rescaling matrix omega, as defined in Banchi et. where c depends on the mode
    al.
    c is a numpy 1D array  of positive floats that controls the amount squeezing required
    alpha is the strength of the weight potentials in the matrix

    returns a 2-d numpy array
    """""
    big_potentials = make_potential_vect()
    omega = c * (np.eye(len(big_potentials)) +alpha * np.diag(big_potentials))
    return omega

def create_cov_mean_alt(Adj,c,alpha,nsubspace,hbar=2,tau=1.1,conv='real'):
    '''
    Create and return the covriance matrix and mean vector used to generate the samples from a GBS experiment
    :param Adj:is the complete adjacency matrix: not necessarily the one used for the sampling since we can take a submatrix with the dimension tuned by n_subspace!!!!
    :param c: is a rescaling coefficient
    :param alpha: is a coefficient that has to be chosen carefully and could introduce a bias in the clique detection

    :param n_subspace: a positive integer for the dimension of the submatrix from the total adjacency matrix to speed-up the sampling
    :param tau: is the flexibility constant used to define the adjacency matrix with the formatting from make_adj.py. The default value for tau is the one used for Tace-As in Banchi et al.
    :param conv: if complex return the outputs in the complex convention in the ordering aadag, else if real returns in the xxpp real convention(the one used by the Walrus!!)
    :return:
    '''
    Id = np.eye(nsubspace)
    weights=make_potential_vect()[:nsubspace]
    Adj=Adj[:nsubspace,:nsubspace]
    omega = make_omega(c, alpha)[:nsubspace,:nsubspace]
    BIG=omega@Adj@omega
    Sigma_Qinv = np.block([[Id, -BIG], [-BIG, Id]])
    Sigma_Q = inv(Sigma_Qinv)
    gamma = np.block([[omega, np.zeros((nsubspace,nsubspace))], [np.zeros((nsubspace,nsubspace)), omega]])@ np.concatenate(((1+1.1*weights)**3,(1+1.1*weights)**3))
    d_alpha=(Sigma_Q @ gamma)[:nsubspace]
    if conv=='real':
        return qt.Covmat(Sigma_Q,hbar=hbar),np.sqrt(2*hbar)*np.concatenate([d_alpha, np.zeros(nsubspace)])


    elif conv=='complex':
        return Sigma_Q-np.eye(2 * nsubspace) / 2,np.sqrt(2*hbar)*np.concatenate([d_alpha,np.conj(d_alpha)])
def samples_cov_alt(Adj,alpha,target_nsqz,n_subspace,nsamples,data_directory,loss_mode=0,hbar=2):
    '''
    Generate samples from the adjacency matrix with the encoding based on BIG=c(1+alpha*weigths)*Adj*c(1+alpha*weigths)

    :param Adj: the complete adjacency matrix of the graph
    :param nsqz_target:  is the target for the mean photon number coming from the squeezing

    :param alpha: is a coefficient that has to be chosen carefully and could introduce a bias in the clique detection
    :param n_subspace: a positive integer for the dimension of the submatrix from the total adjacency matrix to speed-up the sampling
    :param nsamples: the number of samples we want to produce
    :param data_directory:
    :param loss_mode: a float number taking into account total loss of the GBS experiment including: coupling and collection efficiency, transmission, fiber coupling and detection efficiency at each mode at the end of the interferometer
    :param hbar:
    :return: Return a 2D numpy array of samples
    '''
    t=1.-loss_mode
    c=tune_c(alpha,target_nsqz,Adj,n_subspace)
    omega = make_omega(c, alpha)[:n_subspace, :n_subspace]
    BIG = np.dot(np.dot(omega, laplacian(Adj)), omega)
    print("Mean photon number from squeezing:",mean_nsqz(BIG))
    cov_rescaled,mean_rescaled=create_cov_mean_alt(Adj,c,alpha,n_subspace,hbar=hbar) #covariance matrix and mean matrix given the parameters c and v, alpha and Adj
    ncoh=np.sum(np.abs(mean_rescaled)**2)/(2*hbar)# Mean photon number with a mean vector in the xxpp ordering

    print("Mean photon number from displacement:",ncoh)
    path = data_directory + '\\' + 'nsamples={:.1f}'.format(nsamples) + '_nsubspace={:.1f}'.format(n_subspace) + 'alpha={:.1f}'.format(alpha) + 'loss={:.2f}'.format(loss_mode) + 'ncoh={:2f}'.format(ncoh) + '_displaced_samples_cov.csv'

    if loss_mode!=0:
        mu_loss=mean_rescaled.copy()
        cov_loss = cov_rescaled.copy()
        for i in range (n_subspace):
            mu_loss,cov_loss=loss(mu=mu_loss,cov=cov_loss,T=t,nbar=0,mode=i)
        samples = sp.torontonian_sample_state(cov=cov_loss, mu=mu_loss, samples=nsamples,hbar=hbar)

        np.savetxt(path, samples, delimiter=',')
    else:

        samples=sp.torontonian_sample_state(cov=cov_rescaled,mu=mean_rescaled, samples=nsamples)
        np.savetxt(path, samples, delimiter=',')
    return samples,path,ncoh

def mean_nsqz(BIG):
    """

    :param BIG: is the binding interaction graph, a numpy array
    :return: the mean photon number for the squeezing for a normal GBS experiment
    """
    n = 0
    (lambdal_rescaled, U_rescaled) = takagi(BIG)
    for i in range(len(lambdal_rescaled)):
        n+=(lambdal_rescaled[i])**2/(1-(lambdal_rescaled[i])**2)
    return n


def tune_c(alpha,target_nsqz,Adjtot,nsubpsace):
    """

    :param alpha: the alpha at the input of the adjacency matrix
    :param target_nsqz:  positive number representing  target mean photon n for the squeezing
    :param Adjtot:  adjacency matrix of the total graph
    :param nsubpsace:  dimension of the considered subspace
    :return:
    """
    Adj = Adjtot[:nsubpsace, :nsubpsace]
    def cost(c,alpha,target_nsqz,Adj,n_subspace):
        omega = make_omega(c, alpha)[:n_subspace, :n_subspace]
        BIG = np.dot(np.dot(omega, laplacian(Adj)), omega)
        return np.abs(target_nsqz-mean_nsqz(BIG))
    res=minimize_scalar(cost,args=(alpha,target_nsqz,Adj,nsubpsace))
    return res.x