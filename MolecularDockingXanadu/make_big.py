

import csv
import numpy as np
from provide_namespace import gbs_input_ns

# get namespace
ns = gbs_input_ns()

EXP_PATH = "C:/Users/em1120/MolecularDockingXanadu"

def get_my_keys():

    """
    function to retrieve the keys for the BIG matrix, according to the
    formatting produced by make_adj.py

    returns a list of list of strings
    """
    raw_keys = []
    with open(EXP_PATH + "/big/key_tau" + str(ns.tau) + "_.csv",
              newline="") as csvfile:
        keymaker = csv.reader(csvfile, delimiter=";", quotechar="'")
        for row in keymaker:
            raw_keys.append(row)
    raw_keys = raw_keys[0]
    raw_keys.pop()

    # put the keys in list of list format
    list_keys = []
    for key in raw_keys:
        p_type = key[1:4]
        l_type = key[5:-1]
        while p_type[-1].isdigit():
            p_type = p_type[:-1]
        while l_type[-1].isdigit():
            l_type = l_type[:-1]

        list_keys.append([p_type, l_type])

    return list_keys


def make_omega(list_keys, renorm, alpha):
    """
    function to generate the rescaling matrix omega, as defined in Banchi et.
    al.

    returns a 2-d numpy array
    """
    # generate vertex weights

    big_potentials = []
    for pair in list_keys:
        row = potential_key.index(pair[0])
        col = potential_key.index(pair[1])
        big_potentials.append(potential_mat[row, col])


    # generate the rescaling matrix Omega
    # c and alpha are tunable parameters
    # WARNING: they must be carefully chosen.
    omega = renorm * (np.eye(len(big_potentials)) +
                      alpha * np.diag(big_potentials))
    return omega


# define pharmacophore interaction potentials in a matrix
# these are copied straight from table S1 in Banchi et al.

potential_key = ["NC", "PC", "HD", "HA", "Hp", "Ar"]
potential_mat = np.array([
    [0.2953, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.6459, 0.1596, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.7114, 0.4781, 0.5244, 0.0000, 0.0000, 0.0000],
    [0.6450, 0.7029, 0.6686, 0.5478, 0.0000, 0.0000],
    [0.1802, 0.0679, 0.1453, 0.2317, 0.0504, 0.0000],
    [0.0000, 0.1555, 0.1091, 0.0770, 0.0795, 0.1943],
])
potential_mat = potential_mat + np.tril(potential_mat, -1).T

def main(tau, renorm, alpha):
    """
    run the script and save the output
    """
    adj = np.genfromtxt(EXP_PATH + "/big/adj_mat_tau" + str(tau) +
                        "_.csv",
                        delimiter=",")

    omega = make_omega(get_my_keys(), renorm, alpha)

    input_mat = np.dot(omega, np.dot(adj, omega))
    np.savetxt(
        EXP_PATH + "/big/big_tau" + str(ns.tau) + "_.csv",
        input_mat,
        delimiter=",",
    )


if __name__ == "__main__":
    main(ns.tau, ns.c, ns.a)
