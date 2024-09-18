import numpy as np
from Library_Generate_displaced_samples_alternative_encoding_graphs import *
from Library_Analysis import*
from Library_LoopHafnian import*
from time import time
from matplotlib import cm
from matplotlib.ticker import IndexLocator,LinearLocator
import os
import pickle

####Set the working directory in the folder DisplacedGBS_GeneralGraphs
#Graph properties

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

ngraphs=500
nvertices=20
prob_edges=[0.3,0.8]
size_pickle=10 # Number of graph samples per pickle file
index_graph=1 # Index of the graph in the file 
index_file=1 # File number of the graph


#Simulation properties
nsamples=500
nsqz_tot=2
alpha=1
target_nsqz=np.linspace(0.1,1,nsqz_tot)
target_ncoh=5
n_iterations_local_search=7
loss_mode=0.5
data_directory=create_directory()
zero_samples=[] #Record the position of the graph simulation for which all the samples were zero clicks
enhancement_array=np.zeros(size_pickle)
results={'Size_pickle':size_pickle,'Index_file':index_file,'Samples':[], 'EnhancementArray':[],'ZeroSamples':[],'ngraphs':ngraphs,'nvertices':nvertices,'prob_edges':prob_edges,'nsamples':nsamples,'nsqz_tot':nsqz_tot,'alpha':alpha,'target_ncoh':target_ncoh,'target_nsqz':target_nsqz,'niteration_local':n_iterations_local_search,'loss_mode':loss_mode,'succ_gbs':[],'succ_uni':[]}

for s in range(len(target_nsqz)):
    for p in range(len(prob_edges)):
        enhancement_temp=0
        start_all = time()
        for i in range(ngraphs):
            
            if index_graph>size_pickle:

                results['EnhancementArray']=enhancement_array
                if zero_samples!=[]:
                
                    results['ZeroSamples']=zero_samples

                with open(data_directory+'\\Results_{:d}.pickle'.format(index_file),'wb') as handle:
                    pickle.dump(results,handle)

                enhancement_array=np.zeros(size_pickle)
                index_graph=1
                index_file+=1
                results={'Size_pickle':size_pickle,'Index_file':index_file,'Samples':[], 'EnhancementArray':[],'ZeroSamples':[],'ngraphs':ngraphs,'nvertices':nvertices,'prob_edges':prob_edges,'nsamples':nsamples,'nsqz_tot':nsqz_tot,'alpha':alpha,'target_ncoh':target_ncoh,'target_nsqz':target_nsqz,'niteration_local':n_iterations_local_search,'loss_mode':loss_mode,'succ_gbs':[],'succ_uni':[]}


                

            samples, path, ncoh, nsqz, graph_adj, weights= samples_cov_alt(nvertices=nvertices,prob_edges=prob_edges[p],alpha=alpha, target_nsqz=target_nsqz[s],
                                                        target_ncoh=target_ncoh, nsamples=nsamples,
                                                        data_directory=data_directory, loss_mode=loss_mode, hbar=2,
                                                        n_subspace=nvertices)
            results['Samples'].append(samples)
            
            if np.array_equal(samples,np.zeros((nsamples,nsamples))):
                enhancement_array[index_graph-1]=1
                results['ZeroSamples'].append([s,p,index_file,index_graph])
                
            else:

                # np.savetxt('nsamples={:.1f}_nvertices={:.1f}alpha={:.1f}loss={:.1f}ncoh={:.1f}GeneralGraphsDisplacedSamples.txt'.format(nsamples,nvertices,alpha,loss_mode,ncoh),samples)

                # print("Time to create the samples in seconds:", time1)

                # Histogram of the number of photons per sample
                a, nmax = plot_histogram(samples, plot=False)


                #Remove the non-zero clicks event and the collision events (just for PNRS detectors)
                #cleaned_samples, ncollision = clean_samples(samples, nmax)


                # Find the maximum clique with classical algorithm

                clique_max, clique_weight = find_max_clique(graph_adj, weights, networkx_conv=False)
                # print('Max clique', clique_max)
                # print('Weights of the max clique', clique_weight)

                succ_gbs, succ_uni = plot_success_rate_vs_niter(cleaned_GBS_samples=samples, Adj=graph_adj, weights=weights,
                                                                niter=n_iterations_local_search, plot=False)

                succ_sqzcoh_gbs = succ_gbs[-1]
                succ_sqzcoh_uni = succ_uni[-1]
                if np.abs(succ_sqzcoh_uni)<0.0001:
                    succ_sqzcoh_uni=0.001

                enhancement_array[index_graph-1]= succ_sqzcoh_gbs/succ_sqzcoh_uni
                index_graph+=1
                print(i)
                #np.savetxt(data_directory+'\\succ_gbs_ncoh_{:.1f}_nsqz_{:.1f}_nvertices_{:.1f}_probedges_{:.1f}{:d}.txt'.format(ncoh, nsqz,nvertices,prob_edges[p],i), np.array(succ_gbs), delimiter=',')
                #np.savetxt(data_directory+'\\succ_uni_ncoh_{:.1f}_nsqz_{:.1f}_nvertices_{:.1f}_probedges_{:.1f}{:d}.txt'.format(ncoh, nsqz,nvertices,prob_edges[p],i), np.array(succ_uni), delimiter=',')
                results['succ_gbs'].append(succ_gbs)
                results['succ_uni'].append(succ_uni)

                plt.close('all')
        time1 = time() - start_all
        print(time1)



   
results['EnhancementArray']=enhancement_array
results['Size_pickle']=index_graph-1
if zero_samples!=[]:
    #np.savetxt(data_directory+'\\Zero_samples',zero_samples)
    results['ZeroSamples']=zero_samples

with open(data_directory+'\\Results_{:d}.pickle'.format(index_file),'wb') as handle:
    pickle.dump(results,handle)

for key,value in results.items():
    print(key," : ",value)

#with open(data_directory+'\\Results.pickle','rb') as handle:
#    output=pickle.load(handle)

# fig = plt.figure(figsize=plt.figaspect(0.4))
# ax = fig.add_subplot(121)
# for i in range(len(prob_edges)):
#     ax.plot(target_nsqz, enhancement_array[i,:], label=r'edge density={:.1f}'.format(prob_edges[i]))
#
# ax.yaxis.set_major_locator(LinearLocator(numticks=5))
# ax.xaxis.set_major_locator(IndexLocator(base=0.1, offset=0.1))
# ax.legend()
# ax.set_xlabel(r'$\langle n_{nsqz}\rangle$')
# ax.set_ylabel('Enhancement')
# fig.suptitle(r'Enhancement for Erdos-Renyi graphs of {:.1f} vertices and displacement of {:.1f} with {:.1f} graphs'.format(nvertices, target_ncoh, ngraphs), wrap=True)
# plt.savefig('Enhancement for Erdos-Renyi graphs of {:.1f} vertices and displacement of {:.1f} with {:.1f} graphs.pdf'.format(nvertices, target_ncoh, ngraphs),format='pdf')
# fig.show()


