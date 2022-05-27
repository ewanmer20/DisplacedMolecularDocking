
import numpy as np
from strawberryfields.apps import sample

n_mean, n_samples = 8, 500
TAU = 1.1

EXP_PATH = "C:/Users/em1120/MolecularDockingXanadu"

big_mat = np.genfromtxt(EXP_PATH + "/big/big_tau" + str(TAU) +
                        "_.csv",
                        delimiter=",")

gbs_output = sample.sample(big_mat, n_mean, n_samples, loss=0.1, threshold=True)

np.savetxt(EXP_PATH + "/output/gbs_output.csv",
           gbs_output,
           delimiter=",")
