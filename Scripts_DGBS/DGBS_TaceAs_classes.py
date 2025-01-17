from utils import *

import pickle
import pandas as pd
import copy
import os
import matplotlib.pyplot as plt
from time import time  # For runtime of scripts
from strawberryfields.apps import sample, clique
from strawberryfields.apps.sample import postselect

from scipy.sparse.csgraph import laplacian
from scipy.linalg import inv
from scipy.optimize import minimize
import numpy as np
import networkx as nx
import thewalrus.quantum as qt
import thewalrus.samples as sp
from thewalrus.symplectic import loss
import thewalrus as thw



class DGBS_Sampler(): 
    """
    Class for generating samples from a displaced GBS experiment.
    """
    
    def __init__(self,tau=1.1,alpha=2.1, target_nsqz=None, target_ncoh=None,loss_mode=None,hbar=2, n_subspace=24,conv='real',save=False):
        """
        Initialize the simulation with the simulation arguments.
        :param tau: is the flexibility constant used to define the adjacency matrix with the formatting from make_adj.py. The default value for tau is the one used for Tace-As in Banchi et al.
        :param alpha: is a coefficient that has to be chosen carefully and could introduce a bias in the clique detection
        :param target_sqz: target mean photon number for squeezing that needs to be optimized independently from the squeezing
        :param target_ncoh: target mean photon number for displacement that needs to be optimized independently from the squeezing
        :param nsamples: number of samples to generate
        :param loss_mode: is the loss mode for the simulation
        :param hbar: is the value of the reduced Planck constant
        :param n_subspace: a positive integer for the dimension of the submatrix from the total adjacency matrix to speed-up the sampling
        :param conv: if complex return the outputs in the complex convention in the ordering aadag, else if real returns in the xxpp real convention(the one used by the Walrus!!)
    
        Args:
            
        """
        

    #  Adj:is the complete adjacency matrix: not necessarily the one used for the sampling since we can take a submatrix with the dimension tuned by n_subspace!!!!
        Adj,_=make_adj(tau)
        self.data_directory=create_directory()
        self.Adj = Adj
        self.alpha=alpha
        self.target_nsqz=target_nsqz
        self.target_ncoh=target_ncoh
        
        self.hbar=hbar
        self.loss_mode=loss_mode
        self.n_subspace=n_subspace
        self.conv=conv

        self.nsamples=None

        self.c=None
        self.mean_photon_sqz=None
        self.mean_photon_displacement=None
        self.covariance_matrix=None
        self.mean_displacement_vector=None
        self.weights=None
        self.omega=None
        self.BIG=None

        self.result_dic=None
        self.save=save

    def create_cov_mean_alt_phasenoise(self,custom_phase=[]):
        """
        Create and return the covriance matrix and mean vector used to generate the samples from a GBS experiment assuming uniform phase noisy on the input squeezed states
       
        :param c: is a rescaling coefficient
        
        
       
        :param phase_noise: is the standard deviation of the phase noise added to the tanh of the eigenvalues of the BIG matrix
        :return:
        """
        Id = np.eye(self.n_subspace)
        self.weights=make_potential_vect()[:self.n_subspace]
        Adj=self.Adj[:self.n_subspace,:self.n_subspace]
        self.omega = make_omega(self.c, self.alpha)[:self.n_subspace,:self.n_subspace]
        self.BIG=self.omega@laplacian(Adj)@self.omega
        Sigma_Qinv = np.block([[Id, -np.conj(self.BIG)], [-self.BIG, Id]])
        Sigma_Q = inv(Sigma_Qinv)
        params=optimize_displacement(Sigma_Q=Sigma_Q,target_ncoh=self.target_ncoh,omega=self.omega,weights=self.weights,nsubspace=self.n_subspace,hbar=self.hbar).x
        gamma=give_gamma(kappa=params[0],delta=params[1],omega=self.omega,weights=self.weights,nsubspace=self.n_subspace)
        
        if custom_phase==[]:
            d_alpha=(Sigma_Q @ gamma)[:self.n_subspace]
        else:
           d_alpha=(Sigma_Q @ gamma)[:self.n_subspace]*custom_phase

        if self.conv=='real':
            self.covariance_matrix=qt.Covmat(Sigma_Q,hbar=self.hbar)
            self.mean_displacement_vector=np.sqrt(2*self.hbar)*np.concatenate([[d.real for d in d_alpha],[d.imag for d in d_alpha]])


        elif self.conv=='complex':
           self.covariance_matrix=Sigma_Q-np.eye(2 * self.n_subspace) / 2
           self.mean_displacement_vector=np.sqrt(2*self.hbar)*np.concatenate([d_alpha,np.conj(d_alpha)])
    def return_mean_photon_displacement(self):
           """
            Return the mean photon number from the displacement vector.
            """
           if self.conv=='real':
               return np.dot(self.mean_displacement_vector, self.mean_displacement_vector)
           elif self.conv=='complex':
               return np.dot(self.mean_displacement_vector, np.conjugate(self.mean_displacement_vector))/2
        
    def run_sampler(self,nsamples,foldername=datetime.now().strftime("%d-%b-%Y-(%H.%M.%S.%f)"),custom_phase=[]):
        """
        Run the sampler with the given parameters.
    
        Args:
            nsamples (int): Number of samples to generate.
            foldername (str): Name of the folder to save the results.
            custom_phase (numpy array): List of custom phases to apply to the displacement vector.
        
        Returns:
            dict: Dictionary containing the simulation output.
        """
        self.nsamples = nsamples
        t=1.-self.loss_mode
        self.c=tune_c(self.alpha,self.target_nsqz,self.Adj,self.weights,self.n_subspace)
        self.omega = make_omega(self.c, self.alpha)[:self.n_subspace, :self.n_subspace]
        self.BIG = np.dot(np.dot(self.omega, laplacian(self.Adj)), self.omega)
        self.mean_photon_sqz = mean_nsqz(self.BIG)
        

        if self.covariance_matrix is None or self.mean_displacement_vector is None:
            self.create_cov_mean_alt_phasenoise(custom_phase=custom_phase)
        
        self.mean_photon_displacement = self.return_mean_photon_displacement()
        if self.loss_mode!=0:
           mu_loss=self.mean_displacement_vector.copy()
           cov_loss = self.covariance_matrix.copy()
           for i in range (self.n_subspace):
            mu_loss,cov_loss=loss(mu=mu_loss,cov=cov_loss,T=t,nbar=0,mode=i)
            samples = sp.torontonian_sample_state(cov=cov_loss, mu=mu_loss, samples=nsamples,hbar=self.hbar)
            
        else:
           samples=sp.torontonian_sample_state(cov=self.covariance_matrix,mu=self.mean_displacement_vector, samples=nsamples)
        self.result_dic={'samples':samples, 'target_nsqz':self.target_nsqz,
        'target_ncoh':self.target_ncoh,'mean_photon_squeezing':self.mean_photon_sqz,'mean_photon_displacement':self.mean_photon_displacement,'nsamples':nsamples,'custom_phase':custom_phase,'loss_mode':self.loss_mode,
        'covariance_matrix':self.covariance_matrix,'mean_displacement_vector':self.mean_displacement_vector}
        if self.save:
            self.SaveData(foldername,self.result_dic)
        else:
            return self.result_dic

    def SaveData(self,foldername,raw_dict):
    # Save raw data in the result path directory as pickle file 
    # Construct the complete path for saving the plot
        result_path=self.data_directory + '\\' + foldername +'samples_cov.csv'
        result_path_pickle = os.path.join(result_path, "data.pkl")
        with open(result_path_pickle, "wb") as file:
                # Dump the data using pickle.dump
                 pickle.dump(raw_dict, file)
        df=pd.DataFrame(raw_dict,index=[0])
        df.to_csv(result_path+'data.csv',index=True)

class PostProcessingSamples():
    """
    Class for postprocessing the samples generated from a displaced GBS experiment.
    """

    def __init__(self,raw_samples,Adj) -> None:
        self.raw_samples=raw_samples # Raw samples from the GBS experiment
        self.Adj=Adj # Adjacency matrix
        _,nmax=self.plot_histogram(plot=False,phot_dist=False)
        self.nmax=nmax # Maximal number of photons in one sample over all the samples
        self.cleaned_samples,_=self.clean_samples() # Cleaned samples without the zero photon events and the collision events
        self.niterations=None # Number of iterations for the local search algorithm
        self.weights=make_potential_vect() # Weights for the nodes of the graph
        
    
    
    def clean_samples(self):
    #Return the GBS samples without the collisions the zero photon events and the ratio of photon in the non-collision free regime over the non-zero samples
    #tot_samples= 2D array for the list of samples
    #n_max= sample with the maximal number of photons returned by plot_histogram
        initial_samples = postselect(self.raw_samples,1,self.nmax)# Discard the zero clicks event
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

    def find_max_clique(self,networkx_conv=False):
    #Find the maximum clique of a graph given the list of weights
    #networkx_conv: Return the max_clique in the networkx convention
    #WARNING: Weights and Adj has to be the same length
    #WARNING: clique_temp is using the clicks convention!
        temp_Adj=copy.deepcopy(self.Adj)
        if len(self.weights)!=len(temp_Adj):
            raise Exception("Weigths and Adj needs the same length")

        for i in range(len(temp_Adj)):
            temp_Adj[i, i] = self.weights[i]
        weighted_graph = nx.Graph(temp_Adj)
        cliques_tot = nx.find_cliques(weighted_graph)
        max_clique_weight_temp=0
        clique_temp=None
        clique_temp_net=None
        for el in cliques_tot:
            clique=np.zeros(len(temp_Adj),dtype=np.float64)
            for ind in el:
                clique[ind]=1.
            clique_weight=sample_weight(clique,self.weights)
            if clique_weight>max_clique_weight_temp:
                clique_temp=clique
                max_clique_weight_temp=clique_weight
                if networkx_conv==True:
                    clique_temp_net=el
        if networkx_conv==False:
            return clique_temp,max_clique_weight_temp
        else:
            return clique_temp_net




    def plot_histogram(self,plot=True,phot_dist=False):
    #Plot the histogram of the photon number distribution and return the histogram given the samples
    #Warning: the format of each samples must be the list of photons measured per mode (click convention). For instance [2,1,0] says 2 photons have been measured in mode 0, 1 in mode 1 and 0 in mode 2
        photon_number=np.array([sum(s) for s in self.raw_samples]).astype(np.int64)
        nmax=int(np.amax(photon_number))
        hist=np.zeros(nmax+1)
        for s in photon_number:
            hist[int(s)]+=1

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


    def plot_success_rate_vs_niter(self,niterations,plot=True):
    # Plot the success rate of the greedy-shrinking/local_search algorithms on samples produced by GBS as a function of the number of iterations.
    #This success rate is compared with the case of uniform samples
    #cleaned_GBS_samples=all the cleaned samples processed after a GBS simulation (no zero photon events and only 0 or 1 in a sample): a 2D numpy array of integers
    #1D numpy array of weigths for each nodes of the graph
    #Plot the figure and save it in Analysis_folder
    #WARNING: click convention for cleaned_GBS_samples!!!
        self.niterations=niterations
        t0 = time()
        if len(self.weights) != len(self.Adj):
            raise Exception("Weigths and Adj needs the same length")
        _, _, photo_dist = self.plot_histogram(plot=False, phot_dist=True)
        samples_uni = [list(np.random.choice(len(self.Adj), np.abs(photo_dist[i]), replace=False)) for i in
                        range(len(self.cleaned_samples))]  # generates uniform samples in the networkx convention
        max_clique_sample_nxconv = self.find_max_clique(networkx_conv=True)  # The maximum clique
        
        graph_ref = nx.Graph(self.Adj)

        cleaned_samples_copy = copy.deepcopy(self.cleaned_samples)
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

        for i in range(1, self.niterations-1):
            searched_GBS = [clique.search(clique=s, graph=graph_ref, iterations=1, node_select=self.weights) for s in searched_GBS]
            searched_GBS = [sample for sample in searched_GBS if is_clique_networkx(sample, max_clique_sample_nxconv) == False]

            succ_rate_GBS.append((len(shrunk_GBS) - len(searched_GBS)) / (len(shrunk_GBS)) * 100)  # Count the occurences of the max clique in the networkx convention
            searched_uni = [clique.search(clique=s, graph=graph_ref, iterations=1, node_select=self.weights) for s in searched_uni]
            searched_uni = [sample for sample in searched_uni if is_clique_networkx(sample, max_clique_sample_nxconv) == False]
            succ_rate_uni.append((len(shrunk_uni) - len(searched_uni)) / (len(shrunk_uni)) * 100)  # Count the occurences of the max clique in the networkx convention

        t1 = time()
        print(t1 - t0)
        print(succ_rate_GBS)

      
        
        if plot==True:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 16))
            ax.plot(np.array(succ_rate_GBS), label='Displaced GBS samples', color='g')
            ax.plot(np.array(succ_rate_uni), label='Uniform samples', color='r')
            # ax.plot(np.array(clique_rate_uni)/len(cleaned_GBS_samples)*100,'r--',label='Uniform samples bound',)
            # ax.plot(np.array(clique_rate_GBS)/len(cleaned_GBS_samples)*100,'g--',label='GBS samples bound')
            ax.set_xlabel('Iteration step of local search algorithm')
            ax.set_ylabel('Success rate (%)')
            plt.legend()
            plt.show()
            return succ_rate_GBS,succ_rate_uni
        else:
            return succ_rate_GBS,succ_rate_uni

    def plot_histogram_clique_values(self,weights,plot=True):
    #Plot the histograms for the different clique values with different number of photons: one histogram is for the uniform samples and the other one is for GBS sample
        for i in range(1,self.nmax+1):
            cleaned_GBS_samples_nphoton=postselect(self.cleaned_samples,i,i)
            if cleaned_GBS_samples_nphoton==[]:
                pass
            else:
                clique_list,_=count_cliques(cleaned_GBS_samples_nphoton,nx.Graph(self.Adj))
                hist=[]
                for j in range(len(clique_list)):
                    if clique_list[j]==True:
                        hist.append(sample_weight(cleaned_GBS_samples_nphoton[j],weights))
                
                plt.hist(hist,bins=10,label="{:.2f}".format(i))
        plt.xlabel("Clique weight")
        plt.ylabel("Normalized probability(%)")
        plt.legend(loc="upper right")
        if plot==True:
            plt.show()
        else:
            pass
              
class OptimizerPhaseDisplacement():
  """
  Class for optimizing the phase displacement parameters for a displaced GBS experiment to maximise the success rate of finding a max clique
  """

  def __init__(self, sim_params):
    """
    Initialize the optimizer with the simulator and objective function.

    Args:
        sampler (GaussianBeamSimulator): The DGBS sampler object.
    """
    self.sim_params = sim_params
    self.sampler = DGBS_Sampler(**sim_params)
    self.optimal_parameters = None
    self.best_score = None
    self.evaluation_count = 0

  def scipy_minimize_optimization(self,initial_parameters,fixed_args_sampler,fixed_args_postprocessing,method='BFGS', **kwargs):
    """
    Performs the optimization using the chosen method.

    Args:
        initial_parameters (dict): Dictionary containing initial parameter values.
        method (str): The optimization method to use.
        fixed_args (tuple): Tuple containing additional fixed arguments for the simulation in mm.
        **kwargs: Additional arguments for the optimization method.
    """

    
    res = minimize(self.objective_wrapper, initial_parameters,args=(fixed_args_sampler,fixed_args_postprocessing),method=method,tol=1e-1,bounds=[(0,2*np.pi)]*self.sampler.n_subspace,options={'maxiter': 10})
    self.optimal_parameters = res.x
    self.best_score = res.fun

  def scipy_minimize_optimization_test(self, initial_parameters,method='BFGS', **kwargs):
    """
    Performs the optimization using the chosen method.

    Args:
        initial_parameters (dict): Dictionary containing initial parameter values.
        method (str): The optimization method to use.
        fixed_args (tuple): Tuple containing additional fixed arguments for the simulation in mm.
        **kwargs: Additional arguments for the optimization method.
    """


    # Just run the sampler once such that the BIG matrix is defined and can be used in the objective function
    result_dic=self.sampler.run_sampler(nsamples=10,foldername='test',custom_phase=[])
    res = minimize(self.objective_wrapper_test, initial_parameters,method=method,bounds=[(0,2*np.pi)]*self.sampler.n_subspace,**kwargs)
    self.optimal_parameters = res.x
    self.best_score = res.fun
   

  def scipy_minimize_optimization_TwoFold(self, initial_parameters, alpha,method='BFGS', **kwargs):
    """
    Performs the optimization using the chosen method.

    Args:
        initial_parameters (dict): Dictionary containing initial parameter values.
        alpha (float): The alpha parameter for the Renyi entropy.
        method (str): The optimization method to use.
        **kwargs: Additional arguments for the optimization method.
    """


    # Just run the sampler once such that the BIG matrix is defined and can be used in the objective function
    result_dic=self.sampler.run_sampler(nsamples=10,foldername='test',custom_phase=[])
    res = minimize(self.objective_wrapper_TwoFold, initial_parameters,args=(alpha),method=method,bounds=[(0,2*np.pi)]*self.sampler.n_subspace,**kwargs)
    self.optimal_parameters = res.x
    self.best_score = res.fun


  def scipy_minimize_optimization_ThreeFold(self, initial_parameters,alpha,method='BFGS', **kwargs):
    """
    Performs the optimization using the chosen method.

    Args:
        initial_parameters (dict): Dictionary containing initial parameter values.
        alpha (float): The alpha parameter for the Renyi entropy.
        method (str): The optimization method to use.
        **kwargs: Additional arguments for the optimization method.
    """


    # Just run the sampler once such that the BIG matrix is defined and can be used in the objective function
    result_dic=self.sampler.run_sampler(nsamples=10,foldername='test',custom_phase=[])
    res = minimize(self.objective_wrapper_ThreeFold, initial_parameters,args=(alpha),method=method,bounds=[(0,2*np.pi)]*self.sampler.n_subspace,**kwargs)
    self.optimal_parameters = res.x
    self.best_score = res.fun

  def objective_wrapper_test(self,phase_parameters):
    """
    Wrapper function to calculate the probability of generating a max clique based on the loop hafnian.

    Args:
        phase_parameters (array): Array containing the parameters for the simulation.
        

    Returns:
        float: The value of the objective function for the given parameters.
    """
    max_clique=max_clique_list(self.sampler.Adj,self.sampler.weights)
    print(max_clique)
    BIG=self.sampler.BIG
    Id = np.eye(self.sampler.n_subspace)
    Sigma_Qinv = np.block([[Id, -np.conj(BIG)], [-BIG, Id]])
    Sigma_Q = inv(Sigma_Qinv)
    d_alpha=self.sampler.mean_displacement_vector*np.concatenate([np.cos(phase_parameters),np.sin(phase_parameters)])
    norm=np.exp(-0.5*np.dot(d_alpha,Sigma_Qinv@d_alpha))/np.sqrt(np.linalg.det(Sigma_Q))
    reduced_BIG=thw.reduction(BIG,max_clique)
    params=optimize_displacement(Sigma_Q=Sigma_Q,target_ncoh=self.sampler.target_ncoh,omega=self.sampler.omega,weights=self.sampler.weights,nsubspace=self.sampler.n_subspace,hbar=self.sampler.hbar).x
    gamma=give_gamma(kappa=params[0],delta=params[1],omega=self.sampler.omega,weights=self.sampler.weights,nsubspace=self.sampler.n_subspace)
    reduced_diag=thw.reduction(gamma,max_clique)
    np.fill_diagonal(reduced_BIG, reduced_diag)
    return (1-norm*thw.hafnian(reduced_BIG,loop=True)**2).real/(norm*thw.hafnian(reduced_BIG,loop=True)**2).real
  
  def objective_wrapper_TwoFold(self,phase_parameters,alpha=1):
    """
    Wrapper function to calculate the probability of generating a max clique based on the 2-fold probability distribution

    Args:
        phase_parameters (array): Array containing the parameters for the simulation.
        

    Returns:
        float: The value of the objective function for the given parameters.
    """
    twoFold_indices=generate_twofoldstatistics(self.sampler.n_subspace)
    twoFold_graph_list=[create_binary_array(twoFold_indices[i],self.sampler.n_subspace) for i in range(len(twoFold_indices))]
    d_alpha=self.sampler.mean_displacement_vector*np.concatenate([np.cos(phase_parameters),np.sin(phase_parameters)])
    probability_list=[thw.threshold_detection_prob(d_alpha,self.sampler.covariance_matrix,twoFold_graph_list[i],self.sampler.hbar) for i in range(len(twoFold_graph_list))]
    if alpha==1:
        return entropy(probability_list)
    else:
        return renyi_entropy(probability_list,alpha)
  
  def objective_wrapper_ThreeFold(self,phase_parameters,alpha=1):
    """
    Wrapper function to calculate the probability of generating a max clique based on the 2-fold probability distribution

    Args:
        phase_parameters (array): Array containing the parameters for the simulation.
        

    Returns:
        float: The value of the objective function for the given parameters.
    """
    twoFold_indices=generate_threefoldstatistics(self.sampler.n_subspace)
    twoFold_graph_list=[create_binary_array(twoFold_indices[i],self.sampler.n_subspace) for i in range(len(twoFold_indices))]
    d_alpha=self.sampler.mean_displacement_vector*np.concatenate([np.cos(phase_parameters),np.sin(phase_parameters)])
    probability_list=[thw.threshold_detection_prob(d_alpha,self.sampler.covariance_matrix,twoFold_graph_list[i],self.sampler.hbar) for i in range(len(twoFold_graph_list))]
    if alpha==1:
        return entropy(probability_list)
    else:
        return renyi_entropy(probability_list,alpha)
    
  def objective_wrapper(self, phase_parameters,fixed_args_sampler,fixed_args_postprocessing,MAX_EVALUATIONS=1e10):
    """
    Wrapper function to run the sampler and calculate the objective function.

    Args:
        parameters (array): Array containing the parameters for the simulation.
        fixed_args_sampler (tuple): Tuple containing additional fixed arguments for the sampler.
        fixed_args_postprocessing (tuple): Tuple containing additional fixed arguments for the postprocessing.

    Returns:
        float: The value of the objective function for the given parameters.
    """
    
    self.evaluation_count += 1
    if self.evaluation_count > MAX_EVALUATIONS:
        raise Exception("Maximum number of function evaluations reached")
    result_dic=self.sampler.run_sampler(nsamples=fixed_args_sampler["nsamples"],foldername=fixed_args_sampler["foldername"],custom_phase=phase_parameters)
    samples=result_dic['samples']
    
    #Postprocessing the samples
    postprocessing=PostProcessingSamples(samples,self.sampler.Adj)
    succ_gbs,_=postprocessing.plot_success_rate_vs_niter(fixed_args_postprocessing["niterations"],plot=False)
    return -succ_gbs[-1]

  def get_optimal_parameters(self):
    """
    Returns the dictionary containing the optimized parameters.

    Returns:
        dict: The dictionary with optimized parameter values or None if not optimized yet.
    """
    return self.optimal_parameters

  def get_best_score(self):
    """
    Returns the best score achieved during optimization.

    Returns:
        float: The best score or None if not optimized yet.
    """
    return self.best_score
  
if __name__ == "__main__":
    # Define the simulation parameters
    sim_params = {
        "tau": 1.1,
        "alpha": 2.1,
        "target_nsqz": 2.34,
        "target_ncoh": 10,
        "loss_mode": 0.0,
        "hbar": 2,
        "n_subspace": 24,
        "conv": "real",
        "save": False,
    }

    # Define the initial parameters for the optimization
    initial_phase = np.exp(1j*np.random.uniform(0, 2 * np.pi, sim_params["n_subspace"]))

    # Define the fixed arguments for the sampler
    fixed_args_sampler = {
        "nsamples": 1000,
        "foldername": "test",
    }

    # Define the fixed arguments for the postprocessing
    fixed_args_postprocessing = {
        "niterations": 7,
    }

    # Define the optimizer
    optimizer = OptimizerPhaseDisplacement(sim_params)

    # Perform the optimization
    # optimizer.scipy_minimize_optimization(initial_phase, fixed_args_sampler, fixed_args_postprocessing)

    # optimizer.scipy_minimize_optimization_test(initial_phase)
    # time1=time()
    # optimizer.scipy_minimize_optimization_TwoFold(initial_phase,alpha=2)
    # time2=time()
    # print('Elapsed time for optimization',time2-time1)

    # time1=time()
    # optimizer.scipy_minimize_optimization_ThreeFold(initial_phase)
    # time2=time()
    # print('Elapsed time for optimization',time2-time1)

    # Retrieve the optimized parameters
    # optimal_parameters = optimizer.get_optimal_parameters()
    # best_score = optimizer.get_best_score()
    # print(f"Optimal parameters: {optimal_parameters}")
    # print(f"Best score: {best_score}")
    # print(len(optimal_parameters))


    # Compare the optimization methods
    
    # results = []
    # methods = ['SLSQP', 'trust-constr', 'COBYLA', 'TNC', 'L-BFGS-B', 'Powell']
    # num_trials = 5
    # for method in methods:
    #     runtimes = []
    #     scores = []
    #     params=[]
    #     succRate=[]
    #     for _ in range(num_trials):
    #         start_time = time()
    #         optimizer.scipy_minimize_optimization_TwoFold(initial_phase,alpha=2)
    #         optimal_parameters = optimizer.get_optimal_parameters()
    #         best_score = optimizer.get_best_score()
    #         end_time = time()

    #         # Run the sampler with the optimal parameters to check if the results are consistent
    #         sampler_check=DGBS_Sampler(**sim_params)
    #         result_dic = sampler_check.run_sampler(nsamples=2000, foldername="test", custom_phase=optimal_parameters.real)
    #         samples = result_dic["samples"]
    #         #Postprocessing the samples
    #         postprocessing=PostProcessingSamples(samples,sampler_check.Adj)
    #         succ_gbs,_=postprocessing.plot_success_rate_vs_niter(fixed_args_postprocessing["niterations"],plot=False)
    #         succRate.append(succ_gbs[-1])
    #         params.append(optimal_parameters)
    #         runtimes.append(end_time - start_time)
    #         scores.append(best_score)
    #         print('Method:', method, 'Runtime:', end_time - start_time, 'succRate:', succRate)
    #     results.append({'method': method, 'runtime': np.mean(runtimes), 'score': np.mean(scores),'params':np.mean(params),'succRate':np.mean(succRate),'std_dev_succRate': np.std(succRate)},)
    # df = pd.DataFrame(results)
    # # Plot histogram of average scores with error bars
    # plt.bar(df['method'], df['succRate'], yerr=df['std_dev_succRate'], alpha=0.5)
    # plt.xlabel('Method')
    # plt.ylabel('Average Score')
    # plt.title('Average Score with Error Bars')
    # plt.xticks(rotation=45)
    # plt.show()

    results=[]
    alpha_list=np.linspace(0,0.999,10)
    num_trials = 5
    for alpha in alpha_list:
        runtimes = []
        scores = []
        params=[]
        succRate=[]
        for _ in range(num_trials):
            start_time = time()
            optimizer.scipy_minimize_optimization_TwoFold(initial_phase,method='L-BFGS-B',alpha=alpha)
            optimal_parameters = optimizer.get_optimal_parameters()
            best_score = optimizer.get_best_score()
            end_time = time()

            # Run the sampler with the optimal parameters to check if the results are consistent
            sampler_check=DGBS_Sampler(**sim_params)
            result_dic = sampler_check.run_sampler(nsamples=2000, foldername="test", custom_phase=optimal_parameters.real)
            samples = result_dic["samples"]
            #Postprocessing the samples
            postprocessing=PostProcessingSamples(samples,sampler_check.Adj)
            succ_gbs,_=postprocessing.plot_success_rate_vs_niter(fixed_args_postprocessing["niterations"],plot=False) 
            succRate.append(succ_gbs[-1])
            params.append(optimal_parameters)
            runtimes.append(end_time - start_time)
            scores.append(best_score)
            print('alpha',alpha,'Runtime:', end_time - start_time, 'succRate:', succRate)
        results.append({'runtime': np.mean(runtimes), 'score': np.mean(scores),'params':np.mean(params),'succRate':np.mean(succRate),'std_dev_succRate': np.std(succRate)},)
    df = pd.DataFrame(results)
    current_dir = os.path.dirname(__file__)
    os.chdir(current_dir)
    df.to_csv('SuccRtate_vs_AlphaRenyiLessThan1.csv', index=False)
    # df=pd.read_csv('my_data.csv')
    # Plot histogram of average scores with error bars
    plt.figure()
    plt.errorbar(alpha_list, df['succRate'], yerr=df['std_dev_succRate'], alpha=0.5)
    plt.xlabel('alpha')
    plt.ylabel('Average Score')
    plt.title('Average Score with Error Bars')
    plt.xticks(rotation=45)
    plt.show()

    
   
    