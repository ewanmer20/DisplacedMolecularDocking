
import argparse


def adj_mat_ns():
    """
    function to return the namespace for make_adj_mat.py
    """
    parser = argparse.ArgumentParser(
        description="Generate and save the "
        "adjacency matrix of the binding interaction graph (BIG)"
        " given two distance matrices, one for the protein and one for the"
        " ligand. All matrices are accompanied by keys to identify "
        "pharmacophore types.")
    parser.add_argument(
        "tau",
        type=float,
        help="positive constant to capture the flexibility of molecules.",
    )
    parser.add_argument(
        "--heu",
        type=str,
        default="none",
        help="option to exclude certain contacts. hadr excludes HA/D<-->AR; "
        "haa excludes HA<-->HA; hdd excludes HD<-->HD; "
        "all excludes all of the above.",
    )
    return parser.parse_args()


def gbs_input_ns():
    """
    function to return the namespace for prep_gbs_input.py
    """
    parser = argparse.ArgumentParser(
        description="Generate and save the "
        "input matrix for the GBS query. This procedure combines the BIG "
        "adjacency matrix with the rescaling matrix, which incorporates "
        "the BIG vertex weights into the GBS query. This script requires the "
        "outputs of make_adj_mat.py to be in /big/")
    parser.add_argument(
        "tau",
        type=float,
        help="positive constant to capture the flexibility of molecules.",
    )
    parser.add_argument(
        "c",
        type=float,
        help="normalization coefficient to ensure the correct "
        "bounds for the spectrum of the input matrix",
    )
    parser.add_argument(
        "a",
        type=float,
        help="positive constant to control the "
        "bias of the GBS samples toward/away from heavy vs. dense subgraphs.",
    )

    return parser.parse_args()
