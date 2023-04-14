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
# from Generate_samples import*
from time import time
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




def plot_histogram(tot_samples,plot=True,phot_dist=False):
    #Plot the histogram of the photon number distribution and return the histogram given the samples
    #Warning: the format of each samples must be the list of photons measured per mode (click convention). For instance [2,1,0] says 2 photons have been measured in mode 0, 1 in mode 1 and 0 in mode 2
    photon_number=np.array([sum(s) for s in tot_samples]).astype(np.int64)
    nmax=np.int(np.amax(photon_number))
    hist=np.zeros(nmax+1)
    for s in photon_number:
        hist[np.int(s)]+=1

    if phot_dist==False:
        if plot==True:
            fig, ax = plt.subplots(figsize=(16, 16))
            X = np.arange(nmax + 1)
            ax.bar(X, hist, color='b', width=1)
            ax.set_xlabel('Photon number')
            ax.set_ylabel('Number of samples')
            ax.set_xticks(X)
            plt.show()
        else:
            pass

        return hist,nmax
    else:
        if plot == True:
            fig, ax = plt.subplots(figsize=(16, 16))
            X = np.arange(nmax + 1)
            ax.bar(X, hist, color='b', width=1)
            ax.set_xlabel('Photon number')
            ax.set_ylabel('Number of samples')
            ax.set_xticks(X)
            plt.show()
        else:
            pass

        return hist, nmax,photon_number


def plot_success_rate_vs_niter(cleaned_GBS_samples,Adj,niter,weights,plot=True):
    # Plot the success rate of the greedy-shrinking/local_search algorithms on samples produced by GBS as a function of the number of iterations.
    #This success rate is compared with the case of uniform samples
    #cleaned_GBS_samples=all the cleaned samples processed after a GBS simulation (no zero photon events and only 0 or 1 in a sample): a 2D numpy array of integers
    #Adj is the adjacency matrix of the graph from which the samples have been generated
    #niter: a positive integer giving the maximum number of iterations of the algorithms
    #1D numpy array of weigths for each nodes of the graph
    #Plot the figure and save it in Analysis_folder
    #WARNING: click convention for cleaned_GBS_samples!!!

    t0 = time()
    if len(weights) != len(Adj):
        raise Exception("Weigths and Adj needs the same length")
    _, _, photo_dist = plot_histogram(cleaned_GBS_samples, plot=False, phot_dist=True)
    print('mean', np.mean(photo_dist))
    print('std', np.std(photo_dist))
    samples_uni = [list(np.random.choice(len(Adj), np.abs(photo_dist[i]), replace=False)) for i in
                   range(len(cleaned_GBS_samples))]  # generates uniform samples in the networkx convention
    max_clique_sample_nxconv = find_max_clique(Adj, weights, networkx_conv=True)  # The maximum clique
    print('max_clique', max_clique_sample_nxconv)

    graph_ref = nx.Graph(Adj)

    cleaned_samples_copy = copy.deepcopy(cleaned_GBS_samples)
    subgraph_GBS = sample.to_subgraphs(cleaned_samples_copy, graph_ref)
    shrunk_GBS = [clique.shrink(s, graph_ref) for s in subgraph_GBS]
    searched_uni = copy.deepcopy(samples_uni)

    shrunk_uni = [clique.shrink(s, graph_ref) for s in searched_uni]
    succ_rate_GBS = [count_clique_occurence_networkx(shrunk_GBS, max_clique_sample_nxconv) / (len(shrunk_GBS)) * 100]  # Comparison
    succ_rate_uni = [count_clique_occurence_networkx(shrunk_uni, max_clique_sample_nxconv) / (len(shrunk_uni)) * 100]

    searched_GBS = [clique.search(clique=s, graph=graph_ref, iterations=1) for s in shrunk_GBS]
    searched_GBS = [sample for sample in searched_GBS if is_clique_networkx(sample, max_clique_sample_nxconv) == False]
    succ_rate_GBS.append((len(shrunk_GBS) - len(searched_GBS)) / (len(shrunk_GBS)) * 100)  # Count the occurences of the max clique in the networkx convention


    searched_uni = [clique.search(clique=s, graph=graph_ref, iterations=1) for s in shrunk_uni]
    searched_uni = [sample for sample in searched_uni if is_clique_networkx(sample, max_clique_sample_nxconv) == False]
    succ_rate_uni.append((len(shrunk_uni) - len(searched_uni)) / (len(shrunk_uni)) * 100)  # Count the occurences of the max clique in the networkx convention


    for i in range(1, niter-1):
        print(i)
        searched_GBS = [clique.search(clique=s, graph=graph_ref, iterations=1, node_select=weights) for s in searched_GBS]
        searched_GBS = [sample for sample in searched_GBS if is_clique_networkx(sample, max_clique_sample_nxconv) == False]

        succ_rate_GBS.append((len(shrunk_GBS) - len(searched_GBS)) / (len(shrunk_GBS)) * 100)  # Count the occurences of the max clique in the networkx convention
        searched_uni = [clique.search(clique=s, graph=graph_ref, iterations=1, node_select=weights) for s in searched_uni]
        searched_uni = [sample for sample in searched_uni if is_clique_networkx(sample, max_clique_sample_nxconv) == False]
        succ_rate_uni.append((len(shrunk_uni) - len(searched_uni)) / (len(shrunk_uni)) * 100)  # Count the occurences of the max clique in the networkx convention

    t1 = time()
    print(t1 - t0)
    print(succ_rate_uni)
    print(succ_rate_GBS)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 16))
    ax.plot(np.array(succ_rate_GBS), label='GBS samples networkx', color='g')
    ax.plot(np.array(succ_rate_uni), label='Uniform samples', color='r')
    # ax.plot(np.array(clique_rate_uni)/len(cleaned_GBS_samples)*100,'r--',label='Uniform samples bound',)
    # ax.plot(np.array(clique_rate_GBS)/len(cleaned_GBS_samples)*100,'g--',label='GBS samples bound')
    ax.set_xlabel('Iteration step of local search algorithm')
    ax.set_ylabel('Success rate (%)')
    plt.legend()
    if plot==True:
        plt.show()
        return succ_rate_GBS,succ_rate_uni
    else:
        return succ_rate_GBS,succ_rate_uni

def plot_histogram_clique_values(cleaned_GBS_samples,nmax,Adj,weights,plot=True):
    #Plot the histograms for the different clique values with different number of photons: one histogram is for the uniform samples and the other one is for GBS sample
    for i in range(1,nmax+1):
        cleaned_GBS_samples_nphoton=postselect(cleaned_GBS_samples,i,i)
        if cleaned_GBS_samples_nphoton==[]:
            pass
        else:
            clique_list,_=count_cliques(cleaned_GBS_samples_nphoton,nx.Graph(Adj))
            hist=[]
            for j in range(len(clique_list)):
                if clique_list[j]==True:
                    hist.append(sample_weight(cleaned_GBS_samples_nphoton[j],weights))
            print(len(hist))
            plt.hist(hist,bins=10,label="{:.2f}".format(i))
    plt.xlabel("Clique weight")
    plt.ylabel("Normalized probability(%)")
    plt.legend(loc="upper right")
    if plot==True:
        plt.show()
    else:
        pass







