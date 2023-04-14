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
from scipy.optimize import minimize_scalar, Bounds
import matplotlib.pyplot as plt
import copy
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

def make_adj(tau):
    """
    function to create the adjacency matrix for the binding interaction graph (BIG)

    returns the matrix with a key containing the index labels
    """

    # the vertex set of the binding interaction graph (BIG)
    # all pharmacophores are indexed by integers
    # v_set contains lists of lists
    # in the format: [[protein point index, ligand point index]]


    ligand_dists, pocket_dists, ligand_key, pocket_key = get_data()

    v_set = [[i, j] for i in range(len(ligand_key))
             for j in range(len(pocket_key))]

    big_key = [
        "(" + str(ligand_key[vertex[0]]) + "," +
        str(pocket_key[vertex[1]]) + ")" for vertex in v_set
    ]
    big_matrix = fill_mat(ligand_dists, pocket_dists, v_set, tau)

    return big_matrix, big_key

def fill_mat(ligand_dists, pocket_dists, v_set, tau):
    """
    convenience function to fill in the adj matrix
    tau determines the flexibility threshold.
    """
    big_matrix = np.zeros((len(v_set), len(v_set)))
    for row in range(len(big_matrix)):
        for col in range(len(big_matrix)):
            l_dist = ligand_dists[v_set[row][0], v_set[col][0]]
            p_dist = pocket_dists[v_set[row][1], v_set[col][1]]
            if np.abs(p_dist - l_dist) < 4 + tau:
                big_matrix[row, col] = 1
                big_matrix[col, row] = 1
    np.fill_diagonal(big_matrix, 0)

    return big_matrix
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
    potential_data=np.array([[0.1943,0.6450,0.7114],[0.6450,0.5478,0.6686],[0.7144,0.6686,0.5244]])
    # potential data given in the table S1 given in Banchi et al.: First coloumn: Aromatic (AR)
    # Second column: Hydrogen-bond acceptor (HA) and in the last column: Hydrogen-bond donor (HD). To change the ordering, one have to change the mapping function and the potential matrix
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
        [0.0, 3.68, 3.77],
        [0.0,0.0,2.24],
        [0.0, 0.0, 0.0]
    ])
    ligand_dists = ligand_dists + ligand_dists.T
    ligand_key = ["AR", "HA", "HD"]


    pocket_dists = np.array([
        [0.0,5.48,6.97],
        [0.0, 0.0,2.81],
        [0.0, 0.0, 0.0]
    ])
    pocket_dists = pocket_dists + pocket_dists.T
    pocket_key = ["AR", "HA", "HD"]


    return ligand_dists, pocket_dists, ligand_key, pocket_key


def mapping(pharmacopore):
            if pharmacopore=='AR':
                return 0
            elif pharmacopore=='HA':
                return 1
            elif pharmacopore=='HD':
                return 2
# def make_omega(renorm, alpha):
#     """
#     function to generate the rescaling matrix omega, as defined in Banchi et.
#     al.
#     renorm is a positive scalar that is supposed to control the amount squeezing required
#     alpha is the strength of the weigth potentials in the matrix
#
#     returns a 2-d numpy array
#     """
#
#     big_potentials=make_potential_vect()
#     # generate the rescaling matrix Omega
#     # c and alpha are tunable parameters
#     # WARNING: they must be carefully chosen.
#     omega = renorm * (np.eye(len(big_potentials)) +
#                       alpha * np.diag(big_potentials))
#     return omega


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

def create_cov_mean_alt(Adj,c,alpha,target_ncoh,nsubspace,hbar=2,tau=1.1,conv='real'):
    '''
    Create and return the covriance matrix and mean vector used to generate the samples from a GBS experiment
    :param Adj:is the complete adjacency matrix: not necessarily the one used for the sampling since we can take a submatrix with the dimension tuned by n_subspace!!!!
    :param c: is a rescaling coefficient
    :param alpha: is a coefficient that has to be chosen carefully and could introduce a bias in the clique detection
    :param target_ncoh: target mean photon number for displacement that needs to be optimized independently from the squeezing
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
    params=optimize_displacement(Adjtot=Adj,target_ncoh=target_ncoh,omega=omega,weights=weights,nsubspace=nsubspace,hbar=hbar).x
    gamma=give_gamma(kappa=params[0],delta=params[1],omega=omega,weights=weights,nsubspace=nsubspace)
    # gamma = np.block([[omega, np.zeros((nsubspace,nsubspace))], [np.zeros((nsubspace,nsubspace)), omega]])@ np.concatenate(((1+1.1*weights)**2,(1+1.1*weights)**2))
    d_alpha=(Sigma_Q @ gamma)[:nsubspace]
    if conv=='real':
        return qt.Covmat(Sigma_Q,hbar=hbar),np.sqrt(2*hbar)*np.concatenate([d_alpha, np.zeros(nsubspace)])


    elif conv=='complex':
        return Sigma_Q-np.eye(2 * nsubspace) / 2,np.sqrt(2*hbar)*np.concatenate([d_alpha,np.conj(d_alpha)])

def give_gamma(kappa,delta, omega,weights,nsubspace):
    """

    :param kappa:
    :param delta:
    :param omega:
    :param weights:
    :param nsubspace:
    :return:
    """

    return  np.block([[omega, np.zeros((nsubspace,nsubspace))], [np.zeros((nsubspace,nsubspace)), omega]])@ np.concatenate(((1+delta*weights)**(kappa),(1+delta*weights)**(kappa)))


def samples_cov_alt(Adj,alpha,target_nsqz,target_ncoh,n_subspace,nsamples,data_directory,loss_mode=0,hbar=2):
    '''
    Generate samples from the adjacency matrix with the encoding based on BIG=c(1+alpha*weigths)*Adj*c(1+alpha*weigths)

    :param Adj: the complete adjacency matrix of the graph
    :param nsqz_target:  is the target for the mean photon number coming from the squeezing
    :param taarget_ncoh: target mean photon number for displacement that needs to be optimized independently from the squeezing
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
    cov_rescaled,mean_rescaled=create_cov_mean_alt(Adj,c,alpha,target_ncoh,n_subspace,hbar=hbar) #covariance matrix and mean matrix given the parameters c and v, alpha and Adj
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
    return samples,path,ncoh,mean_nsqz(BIG)

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


def optimize_displacement(target_ncoh,omega,weights,Adjtot,nsubspace,hbar=2):
    """
     :param target_ncoh: the target in terms of mean photon number from displacement
    :param omega: the rescaling matrix
    :param weights: the weights of the nodes
    :return: optimize the parameters of kappa and delta to get mean photon from displacement as close as possible to the target
    """
    Id = np.eye(nsubspace)
    Adj = Adjtot[:nsubspace, :nsubspace]
    BIG = omega @ Adj @ omega
    Sigma_Qinv = np.block([[Id, -BIG], [-BIG, Id]])
    Sigma_Q = inv(Sigma_Qinv)

    def cost(params,target_ncoh,Sigma_Q,weights,nsubspace,hbar=2):
        gamma=give_gamma(params[0],params[1],omega,weights,nsubspace)
        d_alpha = (Sigma_Q @ gamma)[:nsubspace]
        mean_rescaled=np.sqrt(2*hbar)*np.concatenate([d_alpha, np.zeros(nsubspace)])
        ncoh = np.sum(np.abs(mean_rescaled) ** 2) / (2 * hbar)
        return (ncoh-target_ncoh)**2
    res=minimize(cost,args=(target_ncoh,Sigma_Q,weights,nsubspace,hbar),bounds=Bounds([0.1,1.],[np.inf,np.inf]),x0=[1.,1.])
    return res

def select_element(prob_array, index,cuttoff):
    index_temp=copy.deepcopy(index)
    index_temp=index_temp[::-1]
    temp=[cuttoff**(i)*index_temp[i] for i in range(len(index))]
    index_array=np.sum(temp)
    return prob_array[index_array]


def conversion_index(index,numodes):
    index_new = np.zeros(numodes)
    for el in index:
        index_new[el] += 1
    return index_new.astype(int)


def generate_threefoldstatistics(numodes, truncation):
    """

    :param numodes: number of modes of the GBS experiment
    :param truncation: truncation of the Hilbert space for each mode
    :return:
    """
    array_index = []
    for i in range(numodes):
        for j in range(numodes):
            for k in range(numodes):
                if i <= j <= k:
                    array_index.append([i, j, k])
    return array_index


def select_threefoldstatistics(probability_tensor_groundthruth, probability_tensor_experiment, array_index,cutoff, numodes,file_title):
    """

    :param probability_tensor_groundthruth: tensor of probabilities representing the groundtruth
    :param probability_tensor_experiment: tensor of probabilities given by the experiment that we want to compare to the experiment
    :param array_index: array_index is an array of indexes for threefold statistics that can be computed by generate_threefoldstatistics function
    :param file_title: file title for the bar plot as a string of characters
    :return:
    """
    plt.close('all')
    threefold_statistics_groundtruth = []
    threefold_statistics_experiment = []
    threefold_statistics_label = []
    for index in array_index:
        new_index = conversion_index(index,numodes)
        prob_gt = select_element(probability_tensor_groundthruth, new_index,cutoff)
        prob_exp = select_element(probability_tensor_experiment, new_index,cutoff)
        threefold_statistics_groundtruth.append(prob_gt)
        threefold_statistics_experiment.append(prob_exp)
        threefold_statistics_label.append(''.join(map(str, index)))

    fig= plt.figure(figsize=plt.figaspect(0.4))
    plt.bar(threefold_statistics_label, -1*np.array(threefold_statistics_groundtruth), label='groundtruth')
    plt.bar(threefold_statistics_label, threefold_statistics_experiment, label='experiment')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.legend()
    plt.savefig(file_title + '.pdf', format='pdf')
    plt.savefig(file_title + '.png', format='png')
    plt.show()
    plt.pause(200)


def tvd(prob1,prob2):
    """

    :param prob1: A one-dimensional array containing the different probabilities of the first distribution
    :param prob2: A one-dimensional array containing the different probabilities of the same length of the first probability
    :return: the Total variation distance between the two renormalized probability distributions prob1 and prob2
    """
    if len(prob1)==len(prob2):
        prob1_copy=copy.deepcopy(prob1)/(np.sum(prob1))
        prob2_copy=copy.deepcopy(prob2)/(np.sum(prob2))

        return 0.5*np.sum(np.abs(prob1_copy-prob2_copy))
    else:
        print("prob1 and prob2 have to be the same length!")
