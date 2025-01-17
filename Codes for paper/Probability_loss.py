import sys
import time
# Add the Prakash folder to sys.path
# Add the Script_DGBS directory to sys.path
sys.path.append(r'C:\Users\em1120\DisplacedMolecularDocking')

# Import calc_unitary from scripts
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from Scripts_DGBS.probability_max_clique import *

current_dir = os.path.dirname(__file__)
plot_dir = os.path.join(current_dir, 'Plots')
os.makedirs(plot_dir, exist_ok=True)
os.chdir(plot_dir)

Adj=np.array([
  [0, 1, 1, 1, 0, 0],
  [1, 0, 1, 1, 0, 1],
  [1, 1, 0, 1, 0, 0],
  [1, 1, 1, 0, 1, 0],
  [0, 0, 0, 1, 0, 1],
  [0, 1, 0, 0, 1, 0]
])
subgraph_1=np.array([1,1,1,1,0,0])
loss=np.linspace(0,1,20)
gamma_val=0.5
c=0.08
cutoff=3
# start=time.time()
# prob_array=probability_array_DGBS(c, gamma_val, Adj, cutoff=cutoff)
# end=time.time()
# print("MaxCliqueProb",prob_array[tuple([1,1,1,1,0,0])])
# loss_result=[]
# print("Time taken for compute prob_array:", end-start)
# start=time.time()
# for i in range(len(loss)):
#     lossy_prob,_=probability_lossy_DGBS(c=c, gamma_val=gamma_val, loss=loss[i], Adj=Adj,prob_array=prob_array,subgraph=subgraph_1, cutoff=cutoff)
#     loss_result.append(lossy_prob)
# end=time.time()
# print("Time taken for compute lossy_prob:", end-start)
# end=time.time()
# print("Time taken for compute prob_array:", end-start)
loss_result=[]
start=time.time()
for i in range(len(loss)):
    lossy_prob=probability_array_DGBS(c, gamma_val, Adj,loss=loss[i], cutoff=cutoff)[tuple(subgraph_1)]
    loss_result.append(lossy_prob)
    print(f"Loss: {loss[i]:.2f}, Probability: {lossy_prob:.2e}")
end=time.time()
print("Time taken for compute lossy_prob:", end-start)


plt.figure()
plt.plot(loss,loss_result)
plt.xlabel('Loss')
plt.ylabel('Probability')
plt.title('Probability of finding the subgraph with loss')
plt.savefig('Probability_loss.svg')
plt.show()


    
