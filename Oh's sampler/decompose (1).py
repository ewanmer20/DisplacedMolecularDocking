import numpy as np
from scipy.stats import unitary_group
from scipy.linalg import block_diag
from scipy.special import factorial
import sys
from tqdm import tqdm
from itertools import combinations

from thewalrus.quantum import Amat, Qmat, Covmat, is_valid_cov, probabilities, gen_Qmat_from_graph
from thewalrus.symplectic import interferometer, xpxp_to_xxpp
from thewalrus.samples import hafnian_sample_graph_rank_one, hafnian_sample_graph
from thewalrus import perm

from chain_rule import get_samples, get_samples_click, loop_hafnian_batch
import strawberryfields as sf 
import strawberryfields.ops as ops
from loop_hafnian import *

from strawberryfields.apps import data, sample, subgraph, plot
from strawberryfields.decompositions import takagi
import networkx as nx

import random



def Haar_matrix(M):
    return unitary_group.rvs(M)

def sq(r):
    return np.diag([np.exp(2 * r), np.exp(-2 * r)])

def get_H(A_in):
    A = np.copy(A_in)
    N = len(A)
    for i in range(N):
        A[i, i] = np.sum(A[i]) - A[i, i]
    
    a = np.zeros(N)
    B = np.zeros((N, N ** 2))
    
    for j in range(N):
        for i in range(j, N):
            if i == j:
                B[i, N * i + i] = np.sqrt(a[i])
            else:
                B[i, N * i + j] = np.sqrt(A[i, j])
                B[j, N * i + j] = np.sqrt(A[i, j])
    return B, A

def get_AVQD(A_in):
    H, A = get_H(A_in)
    #D = np.linalg.norm(H, axis = 0) ** 2
    D = np.sum(H, axis = 0)
    #V = np.nan_to_num(H / np.linalg.norm(H, axis = 0), 0);
    V = np.nan_to_num(H / D, 0)
    Q = D ** 2 / np.sum(D ** 2)
    return A, V, Q, np.sum(D ** 2)


def get_C_sample(A_in, N, n_samples = 10 ** 3, L = 10 ** 3, weight = False): # N not necesarilly even, i.e., we get 2N photons
    A, V, Q, trD = get_AVQD(A_in)
    M = len(A); K = M ** 2

    if weight == False:
        V_modes = []
        for mode in np.arange(len(Q)):
            V_modes.append(V[:, mode] > 0)
        V_modes = np.array(V_modes, dtype = int)
    
        samples = np.empty((0, M), dtype = int)
        while len(samples) < n_samples:
            modes = np.random.choice(K, p = Q, size = N * L)
            temp_samples = V_modes[modes]
            temp_samples = temp_samples.reshape(L, N, M).sum(axis = 1)

            temp_samples = temp_samples[np.sum(temp_samples > 1, axis = 1) == 0]

            samples = np.concatenate((samples, temp_samples))
            #print("C", len(samples))
    else:
        samples = []
        while len(samples) < n_samples:
            modes = np.random.choice(K, p = Q, size = N)
            temp_samples = np.zeros(M, dtype = int)
            for mode in modes:
                #temp_samples = temp_samples + np.random.multinomial(2, V[:, mode]); # optimize this part
                temp_samples[np.random.choice(np.arange(M), replace = False, p = V[:, mode])] += 1; # optimize this part
            if np.sum(temp_samples > 1) == 0:
                samples.append(temp_samples)
            #print("C", len(samples))
        samples = np.array(samples, dtype = int)
                
    return samples

def get_G_sample(A_in, N, n_samples = 10 ** 3, fix_photon = True):
    rl, U = takagi(A_in)
    nrl = rl / np.max(rl) * 0.9
    A_renorm = U @ np.diag(nrl) @ U.T

    k = N
    
    Q = gen_Qmat_from_graph(A_renorm, k)
    cov = Covmat(Q, hbar = 2)
    
    if fix_photon == True:
        samples = []
        for sample in get_samples(np.zeros(len(cov)), cov, cutoff = 1, n_samples = n_samples * 10 ** 3, max_num = k):
            np.random.seed()
            if np.sum(sample) == N:
                samples.append(sample)
                #if len(samples) % 20 == 0:
                #    print("Q", len(samples))
            if len(samples) == n_samples:
                break
        samples = np.array(samples)

        return samples
    else:
        samples = []
        for sample in get_samples(np.zeros(len(cov)), cov, cutoff = 1, n_samples = n_samples * 10 ** 3, max_num = 16):
            np.random.seed()
            if np.sum(sample) == 0:
                continue;            
            samples.append(sample)
            if len(samples) == n_samples:
                break
        samples = np.array(samples)

        return samples

def get_G_l_sample(A_in, N, n_samples = 10 ** 3, loss = 0.5, fix_photon = True):
    rl, U = takagi(A_in)
    nrl = rl / np.max(rl) * 0.9
    
    A_renorm = U @ np.diag(nrl) @ U.T

    k = N
    
    Q = gen_Qmat_from_graph(A_renorm, k / (1 - loss))
    cov0 = Covmat(Q, hbar = 2)
    cov = (1 - loss) * cov0 + loss * np.eye(len(cov0))
    
    if fix_photon == True:
        samples = []
        for sample in get_samples(np.zeros(len(cov)), cov, cutoff = 1, n_samples = n_samples * 10 ** 3, max_num = k):
            np.random.seed()
            if np.sum(sample) == N:
                samples.append(sample)
                if len(samples) % 20 == 0:
                    print("Q", len(samples))
            if len(samples) == n_samples:
                break
        samples = np.array(samples)

        return samples
    else:
        samples = []
        for sample in get_samples(np.zeros(len(cov)), cov, cutoff = 1, n_samples = n_samples * 10 ** 3, max_num = 16):
            np.random.seed()
            if np.sum(sample) == 0:
                continue;            
            samples.append(sample)
            if len(samples) == n_samples:
                break
        samples = np.array(samples)

        return samples


def get_U_sample2(M, k, n_samples):
    res = []
    for i in range(n_samples):
        k = np.random.randint(1, M + 1)
        res.append(np.random.choice(M, k, replace = False))
    return np.array(res)
    
    

def get_U_sample(M, k, n_samples):
    res = []
    for i in range(n_samples):
        res.append(np.random.choice(M, k, replace = False))
    return np.array(res)

def get_edges(samples, A_in):
    res = []
    for sample in samples:
        mode = np.nonzero(sample)[0]
        res.append(np.sum(A_in[np.ix_(mode, mode)]) / 2)
    return np.array(res)


def get_edges_U(samples, A_in):
    res = []
    for sample in samples:
        res.append(np.sum(A_in[np.ix_(sample, sample)]) / 2)
    return np.array(res)

if __name__=="__main__":
    M = 8
    adj_matrix=np.random.randint(2, size=(M, M))
    adj_matrix = np.triu(adj_matrix, 1) + np.triu(adj_matrix, 1).T
    a=get_G_sample(adj_matrix,2 ,10 )
    print(a)
