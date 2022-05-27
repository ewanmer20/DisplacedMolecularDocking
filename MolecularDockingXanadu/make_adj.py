

import csv
import numpy as np
from provide_namespace import adj_mat_ns


def make_adj(tau):
    """
    function to create the adjacency matrix for the binding interaction graph (BIG)

    returns the matrix with a key containing the index labels
    """

    # the vertex set of the binding interaction graph (BIG)
    # all pharmacophores are indexed by integers
    # v_set contains lists of lists
    # in the format: [[protein point index, ligand point index]]


    ligand_dists, pocket_dists, ligand_key, pocket_key = get_data() 

    v_set = [[i, j] for i in range(len(ligand_key))
             for j in range(len(pocket_key))]

    if ns.heu == "none":
        big_key = [
            "(" + str(ligand_key[vertex[0]]) + "," +
            str(pocket_key[vertex[1]]) + ")" for vertex in v_set
        ]
        big_matrix = fill_mat(ligand_dists, pocket_dists, v_set, tau)
    else:
        heu_v_set = trim_v_set(v_set, pocket_key, ligand_key)
        big_key = [
            "(" + str(ligand_key[vertex[0]]) + "," +
            str(pocket_key[vertex[1]]) + ")" for vertex in heu_v_set
        ]

        big_matrix = fill_mat(ligand_dists, pocket_dists, heu_v_set, tau)
    return big_matrix, big_key


def fill_mat(ligand_dists, pocket_dists, v_set, tau):
    """
    convenience function to fill in the adj matrix
    tau determines the flexibility threshold.
    """
    big_matrix = np.zeros((len(v_set), len(v_set)))
    for row in range(len(big_matrix)):
        for col in range(len(big_matrix)):
            l_dist = ligand_dists[v_set[row][0], v_set[col][0]]
            p_dist = pocket_dists[v_set[row][1], v_set[col][1]]
            if np.abs(p_dist - l_dist) < 4 + tau:
                big_matrix[row, col] = 1
                big_matrix[col, row] = 1
    np.fill_diagonal(big_matrix, 0)

    return big_matrix


def get_data():

    ligand_dists = np.array([
        [0.0, 4.6, 9.1, 9.9],
        [0.0, 0.0, 8.1, 8.4],
        [0.0, 0.0, 0.0, 1.2],
        [0.0, 0.0, 0.0, 0.0],
    ])
    ligand_dists = ligand_dists + ligand_dists.T
    ligand_key = ["HD1", "HA1", "Hp1", "Hp2"]


    pocket_dists = np.array([
        [0.0, 2.8, 4.6, 7.6, 5.9, 11.1],
        [0.0, 0.0, 2.7, 5.1, 3.6, 10.5],
        [0.0, 0.0, 0.0, 3.9, 3.5, 12.0],
        [0.0, 0.0, 0.0, 0.0, 2.2, 10.6],
        [0.0, 0.0, 0.0, 0.0, 0.0, 9.00],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.00],

    ])
    pocket_dists = pocket_dists + pocket_dists.T
    pocket_key = ["HD1", "HD2", "HA1", "HA2", "HA3", "Hp1"]


    return ligand_dists, pocket_dists, ligand_key, pocket_key

def main(tau):
    """
    run the script and save results
    """
    adj, key = make_adj(tau)
    np.savetxt(
        EXP_PATH + "/big/adj_mat_tau" + str(tau) + "_.csv",
        adj,
        delimiter=",",
    )
    with open(EXP_PATH + "/big/key_tau" + str(tau) + "_.csv",
              "w") as csv_file:
        for item in key:
            csv_file.write("%s;" % item)


EXP_PATH = "C:/Users/em1120/MolecularDockingXanadu"
ns = adj_mat_ns()

if __name__ == "__main__":
    main(ns.tau)
