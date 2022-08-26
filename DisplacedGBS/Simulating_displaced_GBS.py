from Generate_displaced_samples import *
from time import time
cwd='big\\big_tau1.1_.csv'
BIG=log_data(cwd)
nsubspace=24
Adj = data.TaceAs().adj[:nsubspace,:nsubspace]
alpha=2.1
target_ncoh=1
start_all=time()
nsamples=100000
sq_min=0.2
sq_max=0.4
loss_mode=0.50
data_directory = create_directory()

c=log_data('Parameters_c_v\\TaceAs\\'+'sqmin={:.1f}'.format(sq_min)+'sqmax={:.1f}'.format(sq_max)+'dim={:.1f}'.format(nsubspace)+'ncoh={:.1f}'.format(target_ncoh)+'alpha={:.2f}'.format(alpha)+'cparameters.csv').reshape((nsubspace,))
v=log_data('Parameters_c_v\\TaceAs\\' + 'sqmin={:.1f}'.format(sq_min) + 'sqmax={:.1f}'.format(sq_max) + 'dim={:.1f}'.format(nsubspace) + 'ncoh={:.1f}'.format(target_ncoh)  +'alpha={:.2f}'.format(alpha)+'vparameters.csv').reshape((nsubspace,))
samples = samples_cov(Adj,c,v,alpha,nsubspace,nsamples,data_directory,loss_mode=0,hbar=2)
time1 = time() - start_all
print(time1)

