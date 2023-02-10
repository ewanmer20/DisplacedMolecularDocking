import numpy as np
from strawberryfields.decompositions import takagi
from Generate_displaced_samples import*
EXP_PATH=os.getcwd()


def return_r(Adj,c,v,alpha):
    """""
    Return the list of squeezing parameter tanh(r) computed from the Tagaki-Autonne decomposition of the BIG matrix
    BIG: 2D numpy array of floats
    tanhr: 1D numpy array of floats tanh(r)
    """""
    nsubspace=len(Adj)
    omega=make_generalized_omega(c,alpha)[:nsubspace,:nsubspace]
    BIG = omega @ Adj @ omega + np.diag(v) @ np.eye(nsubspace)
    tanhr=np.abs(eigvalsh(BIG))
    return np.sort(tanhr)[::-1]


def return_alpha(Adj,c,v,alpha):
    """""
    Return the displacement vector of the Gaussian state in the aadag convention 
    omega: rescaling matrix as a diagonal 2D numpy array of floats 
    A: the adjacency matrix as a real symmetric 2D numpy array of floats
    v: a 1D numpy array of float parameters used in the construction of the rescaling matrix 
    """""
    nsubspace=len(Adj)
    Id=np.eye(nsubspace)
    omega=make_generalized_omega(c,alpha)[:nsubspace,:nsubspace]
    BIG=omega@Adj@omega+np.diag(v)@np.eye(nsubspace)
    Sigma_Qinv=np.block([[Id,-BIG],[-BIG,Id]])
    Sigma_Q=inv(Sigma_Qinv)
    gamma=np.concatenate([np.diag(omega),np.diag(omega)])
    return (Sigma_Q@gamma)[:nsubspace]



def abs_error(a,b):
    return np.sum(np.abs(a-b)**2)
def tune_rescaled_parameters(target_tanhr,target_ncoh,alpha,Adjtot,nsubspace):
    """"
    Return the list of c parameters and v that can encode the adjacency matrix Adjtot of dimension nsubspace in a GBS experiment with the list of squeezing parameters target_r and the mean coherent photon number 
    Adjtot: the adjacency matrix of the total graph as a numpy array of 1 and 0
    nsubspace: the dimension of the considered subspace
    target_tanhr: a 1D numpy array giving the target squeezing parameters lambda=tanh(r) of the GBS experiment
    target_ncoh: a float number giving the mean photon number of your coherent source of your GBS experiment
    alpha is the parameter of the rescaling matrix
    """""
    Adj = Adjtot[:nsubspace, :nsubspace]
    target_tanhr=np.sort(target_tanhr)[::-1]
    def cost(params):
        v=params[:nsubspace]
        c=params[nsubspace:]
        output_alpha=return_alpha(Adj,c,v,alpha)
        output_tanhr=return_r(Adj,c,v,alpha)
        relative_error_displacement=((np.sum(output_alpha[:nsubspace]**2)-target_ncoh))**2
        # relative_error_displacement=0
        relative_error_r=abs_error(target_tanhr,output_tanhr)
        if any(np.abs(output_tanhr)>1):
            return max(output_tanhr) ** 2 * nsubspace  # Penalise tanhr larger than 1

        return relative_error_r+(1e-3)*relative_error_displacement

    omega_init = make_generalized_omega(np.ones(nsubspace), alpha)
    BIG_init = (omega_init @ Adj @ omega_init)[:nsubspace, :nsubspace]
    w_max=np.max(np.abs(eigvalsh(BIG_init)))
    guess_c=np.sqrt(target_tanhr[0]/w_max)
    res = minimize(cost,x0=np.concatenate((np.ones(nsubspace),np.ones(nsubspace)*guess_c)))
    return res

def cost(params):
    v=params[:nsubspace]
    c=params[nsubspace:]
    output_alpha=return_alpha(Adj,c,v,alpha)
    output_tanhr=return_r(Adj,c,v,alpha)
    relative_error_displacement=((np.sum(output_alpha[:nsubspace]**2)-target_ncoh))**2
    relative_error_r=abs_error(target_tanhr,output_tanhr)
    if any(np.abs(output_tanhr)>1):
        return max(output_tanhr) ** 2 * nsubspace  # Penalise tanhr larger than 1

    return relative_error_r+(1e-3)*relative_error_displacement

def photon_dist(cov,d,hbar=2):
    '''

    cov(array): covariance matrix in the real basis
    d(array): mean vector in the real basis
    :return: return an array of the mean photon number
    '''
    # cov_xp=qt.Covmat(cov)
    # d_xp=qt.complex_to_real_displacements(cov)
    return qt.means_and_variances.photon_number_mean_vector(d,cov)
def is_collision(cov,d,epsilon):
    '''

    cov(array): covariance matrix in the complex basis
    d(array): mean vector in the complex basis
    :param epsilon: threshold for which if mean_photon>threshold for at least one mode, the system is in the collision regime
    :return: True if the gaussian state is in the collision regime
    '''
    return np.any(photon_dist(cov,d)>epsilon)


# cwd='big\\adj_mat_tau1.1_.csv'
# Adj=log_data(cwd)
# alpha=1.5
# nsubspace=24
# target_ncohs=[0.001,0.050,0.100,0.500,1,1.5,5,10,50,100]
# sq_min=0.2
# sq_max=0.4
# sq_target = np.random.uniform(low=sq_min, high=sq_max, size=(nsubspace,))
# target_tanhr= np.sort(np.tanh(sq_target))[::-1]
# for target_ncoh in target_ncohs:
#
#
#     res=tune_rescaled_parameters(target_tanhr=target_tanhr,target_ncoh=target_ncoh,alpha=alpha,Adjtot=Adj,nsubspace=nsubspace)
#     v=res.x[:nsubspace]
#     c=res.x[nsubspace:]
#
#     omega_output=make_generalized_omega(c,alpha)
#     output_alpha=return_alpha(Adj,c,v,alpha)
#     output_tanhr=return_r(Adj,c,v,alpha)
#     nphotoncoh=np.sum(output_alpha**2)
#     cost_opt=cost(res.x)
#     cov,mean=create_cov_mean(Adj=Adj,c=c,v=v,alpha=alpha,nsubspace=nsubspace,conv='real')
#     dist=photon_dist(cov,mean)
#     print(np.sum(dist))
#     print(target_ncoh+np.sum(np.divide((target_tanhr)**2,1-(target_tanhr)**2)))
#     print(is_collision(cov,mean,1))
#
#
#     if is_collision(cov,mean,1)==False and cost_opt<1e-5:
#         np.savetxt('Parameters_c_v\\TaceAs\\'+'sqmin={:.1f}'.format(sq_min)+'sqmax={:.1f}'.format(sq_max)+'dim={:.1f}'.format(nsubspace)+'ncoh={:.3f}'.format(target_ncoh)+'alpha={:.2f}'.format(alpha)+'cparameters.csv',c,delimiter=',')
#         np.savetxt('Parameters_c_v\\TaceAs\\' + 'sqmin={:.1f}'.format(sq_min) + 'sqmax={:.1f}'.format(sq_max) + 'dim={:.1f}'.format(nsubspace) + 'ncoh={:.3f}'.format(target_ncoh)  +'alpha={:.2f}'.format(alpha)+'vparameters.csv', v,delimiter=',')
#         np.savetxt('Parameters_c_v\\TaceAs\\' + 'sqmin={:.1f}'.format(sq_min) + 'sqmax={:.1f}'.format(sq_max) + 'dim={:.1f}'.format(nsubspace) + 'ncoh={:.3f}'.format(target_ncoh) +'alpha={:.2f}'.format(alpha)+ 'target_squeezing.csv', target_tanhr,delimiter=',')
#         print("Success for ncoh={:.3f}".format(target_ncoh))
#
