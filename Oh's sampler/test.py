import numpy as np 
from scipy.stats import unitary_group
import strawberryfields as sf 
import strawberryfields.ops as ops
from chain_rule import get_samples, get_samples_click
from thewalrus.quantum import photon_number_mean_vector, mean_clicks
from MIS import MIS_IPS
from MIS_click import ClickMIS



M = 4
eng = sf.Engine(backend='gaussian')
prog = sf.Program(M)
U = unitary_group.rvs(M)

r = 1
eta = 0.9
alpha = 0.1
with prog.context as q:
    for i in range(M):
        ops.Sgate(r) | q[i]
        ops.LossChannel(eta) | q[i]
    ops.Interferometer(U) | q
    
state = eng.run(prog).state

# get wigner function displacement and covariance
mu = state.means()
cov = state.cov()

for sample in get_samples(mu, cov, n_samples=10,max_num=10):
    print(sample)