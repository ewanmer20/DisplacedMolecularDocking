from ClusteringMethods import *


def cost_function(displacement,data):
    nsamples=10
    n_dimension=24
    MinPoints=3
    
    sim_params = {
        "tau": 1.1,
        "alpha": 2.1,
        "target_nsqz": displacement[0],
        "target_ncoh": displacement[1],
        "loss_mode": 0.0,
        "hbar": 2,
        "n_subspace": n_dimension,
        "conv": "real",
        "save": False,
        "arb_graph": True
    }
    GBS_results=[]
    for index,row in data.iterrows():
        if int(row["id"])==10 or int(row["id"])==29 :
            GBS_results.append([0])
        elif int(row["id"])==0: 
            Adj=row["Adj"]
            np.fill_diagonal(Adj, 0)
            clusters=GBS_Based_Clustering(N=nsamples,L=int(sim_params["n_subspace"]/3) ,n_mean=int(sim_params["n_subspace"]/2),  
                                      params_GBS_Sampler=sim_params, Adj=Adj,foldername=row["id"],tinit=n_dimension,weights=np.ones(sim_params["n_subspace"]),MinPoints=MinPoints)
            label_GBS=generate_label_fromGBSclusters(clusters)
            GBS_results.append([density_metric(Adj,label_GBS)])
    
    print(1)
    return -GBS_results[0][0]


if __name__=='__main__':
    current_dir = os.path.dirname(__file__)
    os.chdir(current_dir)
    
    n_dimension=24
    filename_adj_df='dataset\\adjacency_matrix_generated_dataset'

    with open('clusters_results.pkl', 'rb') as f:
        cluster_results_dataframe = pickle.load(f)
    
    res = minimize(cost_function,[int(n_dimension/2),5],args=(cluster_results_dataframe),bounds=[(0, 15),(0,15)], options={'maxiter': 1000, 'disp': True})
    print(res.x)