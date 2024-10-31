from ClusteringMethods import *
from math import isnan

def cost_function(hyperparam,data):
    DBSCAN_results=[]
    for index,row in data.iterrows():
        if int(row["id"])==10 or int(row["id"])==29 :
            DBSCAN_results.append([0])
        else: 
            Adj=row["Adj"]
            np.fill_diagonal(Adj, 0)
            # Coordinates of the points to be clustered
            data=np.array([row["x"],row["y"]]).T
            # DBSCAN
            dbscan = DBSCAN(eps=hyperparam[0], min_samples=3).fit(data)
            labels_dbscan=dbscan.labels_
            if isnan(density_metric(Adj,labels_dbscan)):
                DBSCAN_results.append([0])
            else:
                DBSCAN_results.append([density_metric(Adj,labels_dbscan)])
    print(1)
    return -np.mean(np.array(DBSCAN_results))*10


if __name__=='__main__':
    current_dir = os.path.dirname(__file__)
    os.chdir(current_dir)
    nsamples=10
    n_dimension=24
    MinPoints=3
    
    sim_params = {
        "tau": 1.1,
        "alpha": 2.1,
        "target_nsqz": int(n_dimension/2), #From the paper as well 
        "target_ncoh": 0,
        "loss_mode": 0.0,
        "hbar": 2,
        "n_subspace": n_dimension,
        "conv": "real",
        "save": False,
        "arb_graph": True
    }

    filename_adj_df='dataset\\adjacency_matrix_generated_dataset'

    with open('clusters_results.pkl', 'rb') as f:
        cluster_results_dataframe = pickle.load(f)
    
    res = minimize(cost_function,1000,args=(cluster_results_dataframe),bounds=[(0, None)], options={'maxiter': 1000, 'disp': True})
    print(res.x)
    

