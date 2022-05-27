

import csv
import numpy as np
import networkx as nx
import pprint
from strawberryfields.apps import sample, clique, plot, data

tau = 1.1

EXP_PATH = "C:/Users/em1120/MolecularDockingXanadu"
big_mat = np.genfromtxt(EXP_PATH + 
                        '/big/big_tau' + str(tau) + '_.csv', 
                          delimiter=',')

test_sample = np.genfromtxt(EXP_PATH + 
                            "/output/gbs_output.csv", 
                           delimiter=',')

g = nx.Graph(big_mat)

sam = sample.to_subgraphs(test_sample, g)

cliques = [clique.shrink(i, g) for i in sam]
n_iter = 20
cliques = [clique.search(c, g, n_iter) for c in cliques]

cliques = sorted(cliques, key=len, reverse=True)
with open('cliques.csv', 'w') as result_file:
    for item in cliques:
        result_file.write('%s,' % item + "\n")


clique_sizes = [len(i) for i in cliques]
average_clique_size = np.mean(clique_sizes)
print(average_clique_size)

with open('cliques.csv') as fileObj:
     dataStr = fileObj.readlines()

# print(dataStr)
# print(dataStr[0])
# print(dataStr[0][1:-3])

cliques_processed = []
result = {}
k = 0
for i in dataStr:
    # print(i[1:-3])
    temp = []
    temp = i[1:-3].split(',')
    # print(temp)
    temp2 = []
    for j in temp:
        temp2.append(int(j))
    # print(temp2)
    if temp2 not in cliques_processed:
        cliques_processed.append(temp2)
        result[k] = 1
        k += 1
    else:
        x = cliques_processed.index(temp2)
        result[x] += 1

# print(result)

def calculate_weight(v):
    weight = 0
    for i in v:
        if i in [0,1]:
            weight += 0.5244
        elif i in [2,3,4,6,7]:
            weight += 0.6686
        elif i in [11,14,15,16,20,21,22]:
            weight += 0.2317
        elif i in [8,9,10]:
            weight += 0.5478
        elif i in [5,12,13,18,19]:
            weight += 0.1453
        elif i in [17,23]:
            weight += 0.0504
    return weight

cliques = []
for i in range(k):
    cliques.append((cliques_processed[i],result[i],calculate_weight(cliques_processed[i])))

cliques = cliques
cliques = sorted(cliques, key=lambda cliques: cliques[2], reverse=True)

pprint.pprint(cliques)

