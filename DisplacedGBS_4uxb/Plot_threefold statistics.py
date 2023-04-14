import copy

import matplotlib.pyplot as plt
from time import time

import numpy as np
from matplotlib.ticker import IndexLocator,LinearLocator,FixedLocator
from Generate_displaced_samples_alternative_encoding_4uxb import *
import os

def select_element(tensor,index):
    return tensor[index[0]][index[1]][index[2]][index[3]][index[4]][index[5]][index[6]][index[7]][index[8]]

def conversion_index(index):
    index_new=np.zeros(len(index))
    for el in index:
        index_new[el]+=1
    return index_new

def generate_threefoldstatistics(numodes ,truncation):
    """

    :param numodes: number of modes of the GBS experiment
    :param truncation: truncation of the Hilbert space for each mode
    :return:
    """
    array_index =[]
    for i in range(numodes):
        for j in range(numodes):
            for k in range(numodes):
                if i<= j <= k and i<truncation and j<truncation and k<truncation:
                    array_index.append([i, j, k])
    return array_index

def select_threefoldstatistics(probability_tensor_groundthruth,probability_tensor_experiment,array_index,file_title):
    """

    :param probability_tensor_groundthruth:
    :param probability_tensor_experiment:
    :param array_index:
    :param file_title:
    :return:
    """
    threefold_statistics_groundtruth=[]
    threefold_statistics_experiment=[]
    threefold_statistics_label=[]
    for index in array_index:
        new_index=conversion_index(index)
        prob_gt=select_element(probability_tensor_groundthruth,new_index)
        prob_exp=select_element(probability_tensor_experiment,new_index)
        threefold_statistics_groundtruth.append(prob_gt)
        threefold_statistics_experiment.append(prob_exp)
        threefold_statistics_label.append(''.join(map(str,index)))

    fig,ax=plt.figure(figsize=plt.figaspect(0.4))
    ax.bar(threefold_statistics_label,-1.0*threefold_statistics_groundtruth,label='groundtruth')
    ax.bar(threefold_statistics_label,threefold_statistics_experiment,label='experiment')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(file_title+'.pdf',format='pdf')
    plt.savefig(file_title+'.png',format='png')
    plt.show()
    plt.pause(200)


