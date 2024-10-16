from datetime import datetime  # For current day and time
from datetime import date
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import copy
from scipy.optimize import minimize, Bounds,minimize_scalar
from scipy.sparse.csgraph import laplacian
from strawberryfields.apps import clique
from strawberryfields.decompositions import takagi
import networkx as nx



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
    function to generate the potential matrix given the potential value in Banchi et al. using the same formatting for Tace-As
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

def clicks_distribution_to_networkx(samples):
    #Convert the raw samples that are 1D array of 0 and 1 to array with the indexes of the modes where a click has been detected(convention used by networks and Strawberryfields!!)
    converted_samples=[]
    for s in samples:
        converted_samples.append(np.where(s==1.)[0])
    return np.array(converted_samples,dtype=object)

def networkx_distribution_to_clicks(samples,nmodes):
    #Convert from networks convention listing the index of the nodes of the subgraph to a click convention
    #samples: 2DArray of samples with integers in the networkx convention
    #nmodes: integer of the number of modes of the GBS experiment
    converted_samples=[]
    try:
        for s in samples:
            new=np.zeros(nmodes)
            for i in s:
                new[i]=1
            converted_samples.append(new)
        return np.array(converted_samples,dtype=object)
    except IndexError:
        print('Index of the subgraph''s node greater than the number of modes available')
        pass


def sample_weight(sample,weight):
    #Given a sample and the list of weights per nodes, return the weight of the sample
    #sample=a 1D array for the sample. WARNING: Collision-free regime assumed (only 0 or 1 in the sample) and click convention
    # list of weights used to build the BIG matrix with the same length as the sample
    return np.sum(np.multiply(sample,weight))

def count_cliques(list_samples,graph_ref):
    # Return the number of cliques in a list of samples
    # list_samples is a 2D numpy array of samples with clicks convention
    # Graph ref is the adjacency graph from which the samples have been generated
    samples=clicks_distribution_to_networkx(list_samples)
    boolean_list=[clique.is_clique(graph_ref.subgraph(s)) for s in samples]
    return boolean_list,sum(boolean_list)

def count_clique_occurence(list_samples,clique):
    #Count the number of times where a clique occurs in the list of samples
    #list_samples is the list of samples considered
    # clique is the clique we are considering
    #WARNING: list_samples and clique have to be encoded the same way. Better to use click convention since find_max_clique output is using click convention
    count=0
    for s in list_samples:
        if np.sum(np.abs(s-clique))<0.01:
            count+=1
    return count
def is_clique_networkx(sple,clque):
    """
    :param subgraph: 1D numpy array of integers representing the labels of the nodes of the subgraph
    :param graph_clique: 1D numpy array of integers representing the labels of the nodes of the subgraph
    :return: return if sample is equal to the clique in the networkx convention
    """
    if len(sple) == len(clque) and (np.sort(sple) == np.sort(clque)).all():
        return True
    else:
        return False

def max_clique_list(adj,weights=None):
    """
    Return the maximum clique of the graph in the clicks convention
    :param adj:
    :param weights:
    :return:
    """
    if weights is None:
        weights=np.ones(len(adj))
    clique_max_seq, clique_weight = find_max_clique(adj, weights, networkx_conv=False)
    return clique_max_seq.astype(np.int64)

def find_max_clique(Adj,weights,networkx_conv=False):
    #Find the maximum clique of a graph given the list of weights
    #Adj: adjacency matrix of the considered graph (a numpy 2D array with off-diagonal elements either 0 or 1, null on-diagonal elements)
    #Weights: 1D numpy array of weigths for each nodes of the graph
    #networkx_conv: Return the max_clique in the networkx convention
    #WARNING: Weights and Adj has to be the same length
    #WARNING: clique_temp is using the clicks convention!
    temp_Adj=copy.deepcopy(Adj)
    if len(weights)!=len(temp_Adj):
        raise Exception("Weigths and Adj needs the same length")

    for i in range(len(temp_Adj)):
        temp_Adj[i, i] = weights[i]
    weighted_graph = nx.Graph(temp_Adj)
    cliques_tot = nx.find_cliques(weighted_graph)
    max_clique_weight_temp=0
    clique_temp=None
    clique_temp_net=None
    for el in cliques_tot:
        clique=np.zeros(len(temp_Adj),dtype=np.float64)
        for ind in el:
            clique[ind]=1.
        clique_weight=sample_weight(clique,weights)
        if clique_weight>max_clique_weight_temp:
            clique_temp=clique
            max_clique_weight_temp=clique_weight
            if networkx_conv==True:
                clique_temp_net=el
    if networkx_conv==False:
        return clique_temp,max_clique_weight_temp
    else:
        return clique_temp_net



def count_clique_occurence_networkx(list_samples,clque):
    # Count the number of times where a clique occurs in the list of samples
    # list_samples is the list of samples considered
    # clique is the clique we are considering
    # WARNING: Assuming networkx convention for each arguments!
    count=0
    for s in list_samples:
        if len(s) == len(clque) and (np.sort(s) == np.sort(clque)).all():
            count += 1

    return count

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

def make_potential_vect():
    """
    function to generate the potential matrix given the potential value in Banchi et al. using the same formatting for Tace-As
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

def optimize_displacement(target_ncoh,Sigma_Q,omega,weights,nsubspace,hbar=2):
        """
        :param target_ncoh: the target in terms of mean photon number from displacement
        :param omega: the rescaling matrix
        :param weights: the weights of the nodes
        :return: optimize the parameters of kappa and delta to get mean photon from displacement as close as possible to the target
        """

        def cost(params,target_ncoh,Sigma_Q,weights,nsubspace,hbar=2):
            gamma=give_gamma(params[0],params[1],omega,weights,nsubspace)
            d_alpha = (Sigma_Q @ gamma)[:nsubspace]
            mean_rescaled=np.sqrt(2*hbar)*np.concatenate([d_alpha, np.zeros(nsubspace)])
            ncoh = np.sum(np.abs(mean_rescaled) ** 2) / (2 * hbar)
            return (ncoh-target_ncoh)**2
        res=minimize(cost,args=(target_ncoh,Sigma_Q,weights,nsubspace,hbar),bounds=Bounds([0.1,1.],[np.inf,np.inf]),x0=[1.,1.])
        return res
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

def entropy(prob):
    """
    :param prob: 1D numpy array of probabilities
    :return: the entropy of the probability distribution
    """
    return -np.sum(prob*np.log(prob))

def generate_twofoldstatistics(numodes):
    """

    :param numodes: number of modes of the GBS experiment
    :return:
    """
    array_index = []
    for i in range(numodes):
        for j in range(numodes):
                if i <= j:
                    array_index.append([i, j])
    return array_index

if __name__ == "__main__":
    print("This is a utility file and cannot be run directly")
    print(generate_twofoldstatistics(4))



