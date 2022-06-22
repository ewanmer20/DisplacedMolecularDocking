import copy
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
import networkx as nx
import csv
import plotly
from Generate_samples import*
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

def clean_samples(tot_samples,n_max):
    #Return the GBS samples without the collisions the zero photon events and the ratio of photon in the non-collision free regime over the non-zero samples
    #tot_samples= 2D array for the list of samples
    #n_max= sample with the maximal number of photons returned by plot_histogram
    initial_samples = postselect(tot_samples,1,n_max)# Discard the zero clicks event
    clean_samples=[]
    length_init=len(initial_samples)
    count=0
    for s in initial_samples:
        if np.max(s)>1.:
            count+=1
            new_s=[]
            for i in range(len(s)):
                if s[i]<2.:
                    new_s.append(s[i])
                else:
                    new_s.append(1)

            clean_samples.append(np.array(new_s))
        else:
            clean_samples.append(s)

    return np.array(clean_samples),count/length_init


def sample_weight(sample,weight):
    #Given a sample and the list of weights per nodes, return the weight of the sample
    #sample=a 1D array for the sample. WARNING: Collision-free regime assumed (only 0 or 1 in the sample)
    # list of weights used to build the BIG matrix with the same length as the sample
    return np.sum(np.multiply(sample,weight))

def count_cliques(list_samples,graph_ref):
    # Return the number of cliques in a list of samples
    # list_samples is a 2D numpy array of samples
    # Graph ref is the adjacency graph from which the samples have been generated
    boolean_list=[clique.is_clique(graph_ref.subgraph(s)) for s in list_samples]
    return sum(boolean_list)
def count_clique_occurence(list_samples,clique):
    count=0
    for s in list_samples:
        if np.sum(np.abs(s-clique))<0.01:
            count+=1
    return count

def find_max_clique(Adj,weights):
    #Find the maximum clique of a graph given the list of weights
    #Adj: adjacency matrix of the considered graph (a numpy 2D array with off-diagonal elements either 0 or 1, null on-diagonal elements)
    #Weights: 1D numpy array of weigths for each nodes of the graph
    #WARNING: Weights and Adj has to be the same length

    if len(weights)!=len(Adj):
        raise Exception("Weigths and Adj needs the same length")

    for i in range(len(Adj)):
        Adj[i, i] = weights[i]
    weighted_graph = nx.Graph(Adj)
    cliques_tot = nx.find_cliques(weighted_graph)
    max_clique_weight_temp=0
    clique_temp=None
    for el in cliques_tot:
        clique=np.zeros(len(Adj),dtype=np.float64)
        for ind in el:
            clique[ind]=1.
        clique_weight=sample_weight(clique,weights)
        if clique_weight>max_clique_weight_temp:
            clique_temp=clique
            max_clique_weight_temp=clique_weight
    return clique_temp,max_clique_weight_temp




def plot_histogram(tot_samples):
    #Plot the histogram of the photon number distribution and return the histogram given the samples
    photon_number=np.array([sum(s) for s in tot_samples])
    nmax=np.int(np.amax(photon_number))
    hist=np.zeros(nmax+1)
    for s in photon_number:
        hist[np.int(s)]+=1
    fig,ax=plt.subplots(figsize=(16,16))
    X=np.arange(nmax+1)
    ax.bar(X,hist,color='b',width=1)
    ax.set_xlabel('Photon number')
    ax.set_ylabel('Number of samples')
    ax.set_xticks(X)
    plt.show()
    return hist,nmax

def plot_success_rate_vs_niter(cleaned_GBS_samples,nmax,Adj,niter,weights):
    # Plot the success rate of the greedy-shrinking/local_search algorithms on samples produced by GBS as a function of the number of iterations.
    #This success rate is compared with the case of uniform samples
    #tot_GBS_samples=all the cleaned samples processed after a GBS simulation: a 2D numpy array of integers
    #nmax= a positive integer to postselect all the samples
    #Adj is the adjacency matrix of the graph from which the samples have been generated
    #niter: a positive integer giving the maximum number of iterations of the algorithms
    #1D numpy array of weigths for each nodes of the graph
    #Plot the figure and save it in Analysis_folder
    if len(weights)!=len(Adj):
        raise Exception("Weigths and Adj needs the same length")

    samples_uni = [list(np.random.choice(1,nmax, replace=False)) for i in range(len(cleaned_GBS_samples))] # generates uniform samples
    max_clique_sample=find_max_clique(Adj,weights) #The maximum clique
    graph_ref=nx.Graph(Adj)
    searched_GBS=copy.deepcopy(cleaned_GBS_samples)
    searched_uni=copy.deepcopy(samples_uni)
    succ_rate_GBS=[count_cliques(cleaned_GBS_samples,graph_ref)]
    succ_rate_uni=[count_cliques(samples_uni,graph_ref)]

    for i in range(1,niter):
        shrunk_GBS = [clique.shrink(s, graph_ref) for s in searched_GBS]
        searched_GBS = [clique.search(s, graph_ref,1) for s in shrunk_GBS]
        succ_rate_GBS.append(count_cliques(searched_GBS,graph_ref))

        shrunk_uni = [clique.shrink(s, graph_ref) for s in searched_uni]
        searched_uni = [clique.search(s, graph_ref,1) for s in shrunk_uni]
        succ_rate_uni.append(count_cliques(searched_uni, graph_ref))
    print(succ_rate_uni)
    print(succ_rate_GBS)
    fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(16,16))
    ax.plot(np.array(succ_rate_GBS)/len(cleaned_GBS_samples)*100,label='GBS samples',color='g')
    ax.plot(np.array(succ_rate_uni)/len(cleaned_GBS_samples)*100,label='Uniform samples',color='r')
    ax.set_xlabel('Iteration step of greedy shrinking/local algorithm')
    ax.set_ylabel('Success rate (%)')
    plt.legend()
    plt.show()

# def plot_histogram_clique_values(cleaned_GBS_samples,nmax):
#     #Plot the histograms for the different clique values with different number of photons: one histogram is for the uniform samples and the other one is for GBS sample






