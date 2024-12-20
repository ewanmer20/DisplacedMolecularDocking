from DGBS_ArbitraryGraph_class import *
from sklearn.cluster import KMeans,DBSCAN



def extract_data_from_directory(directory_path):
  """Extracts data from CSV files in the specified directory and stores it in a DataFrame.

  Args:
    directory_path: Path to the directory containing the CSV files.

  Returns:
    A Pandas DataFrame containing the extracted data.
  """

  dataframes = []
  for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
      file_path = os.path.join(directory_path, filename)
      df = pd.read_csv(file_path)

      # Add an ID column with the filename
      df['id'] = filename
      dataframes.append(df)


  # Concatenate all DataFrames into a single DataFrame
  combined_df = pd.concat(dataframes, ignore_index=True)
 
  return combined_df

def create_dictionaries_from_dataframe(df):
  """Creates dictionaries labeled by filenames containing x and y coordinates.

  Args:
    df: A Pandas DataFrame with columns 'x', 'y', and 'id'.

  Returns:
    A dictionary where keys are filenames and values are lists of (x, y) coordinate tuples.
  """

  dictionaries = {}
  for filename, group in df.groupby('id'):
    coordinates = group[['x', 'y']].values.tolist()
    dictionaries[filename] = np.array(coordinates)
  return dictionaries


def select_random_nodes_and_reduce_adjacency_matrix(adjacency_matrix, num_nodes):
  """Selects a random subset of nodes from a graph represented by an adjacency matrix and returns the reduced adjacency matrix.

  Args:
    adjacency_matrix: A NumPy array representing the adjacency matrix of the graph.
    num_nodes: The number of nodes to select.

  Returns:
    A tuple containing a list of selected node indices and the reduced adjacency matrix.
  """

  # Get the number of nodes in the graph
  num_total_nodes = adjacency_matrix.shape[0]

  # Generate a random sample of node indices
  selected_nodes = np.random.choice(num_total_nodes, size=num_nodes, replace=False)

  # Create a boolean mask for the selected nodes
  node_mask = np.zeros(num_total_nodes, dtype=bool)
  node_mask[selected_nodes] = True

  # Reduce the adjacency matrix to the selected nodes
  reduced_adjacency_matrix = adjacency_matrix[node_mask][:, node_mask]

  return selected_nodes, reduced_adjacency_matrix

def computeDistanceMatrix(data):
    """
    Compute the distance matrix of the data
    :param data: the data
    :return: the distance matrix

    """
    return [[np.linalg.norm(data[i]-data[j]) for j in range(len(data))] for i in range(len(data))]


def buildAdjacencyMatrix(D,d_tilda):
    """
    Build the adjacency matrix of the graph
    :param D: the distance matrix
    :param d_tilda: the threshold
    :return: the adjacency matrix

    """
    return np.array([[1 if np.any(D[i, j] < d_tilda) else 0 for j in range(len(D))] for i in range(len(D))])

def find_indices_of_ones(array):
  """Finds the indices of elements equal to 1 in a NumPy array.

  Args:
    array: A NumPy array.

  Returns:
    A list of indices where the elements are equal to 1.
  """

  indices = np.where(array == 1)[0]
  return indices.tolist()

def reduce_adjacency_matrix(adjacency_matrix, selected_nodes):
  """Reduces an adjacency matrix to only include the selected nodes.

  Args:
    adjacency_matrix: A numpy array representing the adjacency matrix of the graph.
    selected_nodes: A list of 0 and 1, where 1 corresponds to the selected nodes.

  Returns:
    The reduced adjacency matrix.
  """

  return thw.reduction(adjacency_matrix,selected_nodes)

def BigEnough(remaining_nodes,MinPoints):
    """
    Check if the remaining nodes are enough
    :param remaining_nodes: the remaining nodes
    :param MinPoints: the minimum number of points
    :return: True if the remaining nodes are enough, False otherwise
    """
    if np.sum(remaining_nodes)>=MinPoints:
        return True
    else :
        return False
    
def Density(remaining_nodes,Adj):
    """
    Compute the density of the graph
    :param Adj: the adjacency matrix
    :return: the density
    """
    subgraph=reduce_adjacency_matrix(Adj,remaining_nodes)
    n=len(subgraph)
    num_edges=np.sum(subgraph)//2
    max_edges=n*(n-1)//2
    Density=num_edges/max_edges
    return Density


def findDensestCandidate(samples,Adj):
    """
    Find the densest candidate
    :param samples: the samples
    :param Adj: the adjacency matrix
    :return: the densest candidate
    """
    best=None
    dbest=0
    for i in range(len(samples)):
        subgraph=reduce_adjacency_matrix(Adj,samples[i])
        if Density(subgraph,Adj)>=dbest:
            if Density(subgraph,Adj)==dbest:
                if np.sum(samples[i])>np.sum(best):
                    dbest=Density(subgraph,Adj)
                    best=samples[i]
            else:
                dbest=Density(subgraph,Adj)
                best=samples[i]
    return best, dbest

def computeThreshold(t,i):
    """
    Compute the threshold
    :param t: the threshold
    :param i: the iteration number
    :return: the updated threshold
    """
    return t*0.9**i

def removeFoundCluster(remaining_nodes,best):

    """
    Remove the found cluster
    :param remaining_nodes: the remaining nodes
    :param best: the best cluster
    :return: the updated adjacency matrix
    """
    for i in range(len(remaining_nodes)):
        if best[i]==1:
            remaining_nodes[i]=0
    return remaining_nodes 


def updateParameters(remaining_nodes):
    """
    Update the parameters
    : remaining_nodes: the remaining nodes
    :return: the updated parameters L and n_mean
    """
    l=len(remaining_nodes)
    return int(l/3), int(l/2)

def postSelectSamples(samples,L):
    """
    Post-select the samples
    :param samples (array): the samples
    :param L (int): the threshold
    :return: the selected samples"""
    return [samples[i] for i in range(len(samples)) if np.sum(samples[i])>=L]


def postProcessing(clusters,remaining_nodes,Adj):
    """
    Post-process the clusters
    :param clusters: the clusters
    :param remaining_nodes: the remaining nodes
    :param Adj: the adjacency matrix
    :return: the updated clusters
    """

    for i in range(len(remaining_nodes)):
        if remaining_nodes[i]==1:
            reduced_subgraph=reduce_adjacency_matrix(Adj,remaining_nodes)
            if np.sum(reduced_subgraph[0,:])==0:
                subgraph=np.zeros(len(remaining_nodes))
                subgraph[i]=1
                clusters.append(subgraph)
                remaining_nodes[i]=0
            else:
                ratio_connectivity=np.zeros(len(clusters))
                for i in range(len(clusters)):
                    indices_cluster=find_indices_of_ones(clusters[i])
                    reduced_subgraph=reduce_adjacency_matrix(Adj,indices_cluster)
                    ratio_connectivity[i]=np.sum([Adj[i,j]+Adj[j,i] for j in range(len(Adj)) if clusters[i][j]>0])/(2*np.sum(clusters[i]))
                index_cluster=np.argmax(ratio_connectivity)
                clusters[index_cluster][i]=1
    return clusters

def GBS_Based_Clustering(N, L, n_mean,params_GBS_Sampler,Adj,foldername,tinit,weights,MinPoints=3):
    """
    Perform the GBS based clustering
    :param N: the number of samples
    :param L: the threshold
    :param n_mean: the mean number of photons
    :param params_GBS_Sampler: the parameters of the GBS sampler
    :param Adj: Adjacency matrix
    :param foldername: the folder name
    :param tinit: the initial threshold
    :param weights: the weights of the nodes in the graph (if unweighted graph, they should be set to 1)
    :param MinPoints: the minimum number of points
    :return: the clusters
    """
    remaining_nodes=np.ones(len(Adj))
    clusters=[]
    Sampler=DGBS_Sampler(**params_GBS_Sampler)
    Sampler.target_nsqz=n_mean
    counter=0
    while BigEnough(remaining_nodes=remaining_nodes,MinPoints=MinPoints):
        i=0
        Go=True
        t=tinit
        
        while Go:
            result_samples=Sampler.run_sampler(nsamples=N,foldername=foldername,Adj=Adj,weights=weights)
            samples=postSelectSamples(result_samples["samples"],L)
            best, dbest= findDensestCandidate(samples,Adj)
            t=computeThreshold(t,i)
            if dbest>t:
                clusters.append(best)
                Go=False
            i+=1
            
        remaining_nodes=removeFoundCluster(remaining_nodes,best)
        print(remaining_nodes)
        L, n_mean=updateParameters(Adj)
        counter+=1
        # print(f"counter: {counter}")
    remaining_nodes=remaining_nodes.astype(int)
    return postProcessing(clusters,remaining_nodes,Adj)
    

def GBS_Based_Clustering_Alternative(N, L, n_mean,params_GBS_Sampler,Adj,foldername,weights,MinPoints=3):
    """
    Perform the GBS based clustering
    :param N: the number of samples
    :param L: the threshold
    :param n_mean: the mean number of photons
    :param params_GBS_Sampler: the parameters of the GBS sampler
    :param Adj: Adjacency matrix
    :param foldername: the folder name
    :param weights: the weights of the nodes in the graph (if unweighted graph, they should be set to 1)
    :param MinPoints: the minimum number of points
    :return: the clusters
    """
    remaining_nodes=np.ones(len(Adj))
    clusters=[]
    Sampler=DGBS_Sampler(**params_GBS_Sampler)
    Sampler.target_nsqz=n_mean
    counter=0
    while BigEnough(remaining_nodes=remaining_nodes,MinPoints=MinPoints):

        result_samples=Sampler.run_sampler(nsamples=N,foldername=foldername,Adj=Adj,weights=weights)
        samples=postSelectSamples(result_samples["samples"],L)
        best, _= findDensestCandidate(samples,Adj)
        if samples!=[]:
            
            remaining_nodes=removeFoundCluster(remaining_nodes,best)
            L, n_mean=updateParameters(Adj)
        counter+=1
    remaining_nodes=remaining_nodes.astype(int)
    if clusters!=[]:
        return postProcessing(clusters,remaining_nodes,Adj)
    else:
        return clusters

def intra_inter_cluster_cohesion(Adj,labels):
    """
    Compute the intra and inter cluster cohesion
    :param Adj: the adjacency matrix built upon the data
    :param labels: the labels
    :return: the intra and inter cluster cohesion
    """
    # Calculate intra-cluster cohesion
    intra_cohesion = 0
    for cluster_label in set(labels):
        cluster_indices = np.where(labels == cluster_label)[0]
        reduced_Adj=Adj[np.ix_(cluster_indices,cluster_indices)]
        edge_count = int(np.sum(reduced_Adj) / 2)
        intra_cohesion += edge_count/(2*(len(cluster_indices)*len(cluster_indices)/2))
    
    # Calculate inter-cluster cohesion
    inter_cohesion = 0
    for cluster_label in set(labels):
        cluster_indices = np.where(labels == cluster_label)[0]
        outside_cluster_indices = np.where(labels != cluster_label)[0]
        for index in cluster_indices:
            inter_cohesion += np.sum(Adj[index, outside_cluster_indices])/(len(cluster_indices)*(len(Adj)-len(cluster_indices)))           
    return intra_cohesion-inter_cohesion    
    
def density_metric(Adj,labels):
    """
    Compute the density metric
    :param Adj: the adjacency matrix
    :param labels: the labels
    :return: the density metric
    """
    density=0
    for cluster_label in set(labels):
        cluster_indices = np.where(labels == cluster_label)[0]
        reduced_Adj=Adj[np.ix_(cluster_indices,cluster_indices)]
        density=+np.sum(reduced_Adj)/len(cluster_indices)
    return density/len(Adj)
        
def generate_label_fromGBSclusters(clusters):
    """
    Generate the labels from the GBS clusters
    :param clusters: the clusters
    :param Adj: the adjacency matrix
    :return: the labels
    """
    labels=np.zeros(len(clusters[0]))
    for i in range(len(clusters)):
        clusters_i=np.array(clusters[i])
        indices_cluster=np.where(clusters_i>=1)[0]
        labels[indices_cluster]=i
    return labels
    
if __name__ == '__main__':
    # Read the data from archive files
    current_dir = os.path.dirname(__file__)
    os.chdir(current_dir)
    # data=extract_data_from_directory('archive_test\\')
    # data = data.drop("Unnamed: 0.1", axis=1)
    # data.to_csv('data_test.csv', index=False)
    # Generate the Distance Matrix for each dataset and store it in a pd dataframe
    # df=pd.read_csv('data.csv')
    # datasets = df.drop(df[df['id'] == 'wave.csv'].index)
    # datasets_dictionary=create_dictionaries_from_dataframe(datasets)
    # distance_matrix_dataframe=pd.DataFrame({'id': [], 'DistanceMatrix': [],'color': [],'x': [],'y': []})
    # for filename, data in datasets_dictionary.items():
    #     print(f'Computing distance matrix for graph {filename}...')
    #     distance_matrix_dataframe = pd.concat([distance_matrix_dataframe,pd.DataFrame({'id': filename, 
    #                                                                                    'DistanceMatrix': [computeDistanceMatrix(data)], 
    #                                                                                     'color': [datasets[datasets['id'] == filename]['color'].to_numpy()],
    #                                                                                     'x':[datasets[datasets['id'] == filename]['x'].to_numpy()],
    #                                                                                     'y':[datasets[datasets['id'] == filename]['y'].to_numpy()]})], ignore_index=True)
        
  
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

   
    # Load the distance matrix dictionary from the pickle file

    filename_adj_df='dataset\\adjacency_matrix_generated_dataset'

 
    # # Compute the adjacency matrix for each dataset
    
    # adjacency_matrix_dataframe=distance_matrix_dataframe.copy()   
    # adjacency_matrix_list=[]
  
    # for index, row in adjacency_matrix_dataframe.iterrows():
    #     distance_matrix=np.array(row["DistanceMatrix"])
    #     print(f'Distance matrix dimensions for graph {row["id"]}: {distance_matrix.shape}')
    #     selected_nodes, reduced_distance_matrix = select_random_nodes_and_reduce_adjacency_matrix(distance_matrix, n_dimension) # Select a random subset of nodes and reduce the adjacency matrix to a dimension that is suitable for GBS simulation
    #     adjacency_matrix_list.append([buildAdjacencyMatrix(reduced_distance_matrix,np.quantile(reduced_distance_matrix, 0.35, axis=0))]) # Based from the paper, the threshold is the 35th percentile of the distance matrix to build the adjacency matrix
    #     row["color"]=row["color"][selected_nodes] #Reduce the size for the color array to the selected nodes
    #     row["x"]=row["x"][selected_nodes] #Reduce the size for the x array to the selected nodes
    #     row["y"]=row["y"][selected_nodes] #Reduce the size for the y array to the selected nodes
    #     row["DistanceMatrix"]=reduced_distance_matrix  #Reduce the size of the distance matrix to the selected nodes
    #     print('row',row)
      
    #     print(f'Distance matrix dimensions for  reduced graph {row["id"]}: {reduced_distance_matrix.shape}')

        
    # adjacency_matrix_dataframe.insert(1, "Adj", adjacency_matrix_list, True)
    # with open(filename_adj_df+'.pkl', 'wb') as f:
    #     pickle.dump(adjacency_matrix_dataframe, f)
    
    # adjacency_matrix_dataframe.to_csv(filename_adj_df+'.csv', index=False)

    # Load the adjacency matrix dictionary from the pickle file

    with open(filename_adj_df+'.pkl', 'rb') as f:
        adjacency_matrix_dataframe = pickle.load(f)
    
    cluster_results_dataframe = adjacency_matrix_dataframe.copy()
    cluster_list=[]
    parameter_list=[]
    
    cluster_results_filename='clusters_results'
    for index, row in adjacency_matrix_dataframe.iterrows():
      print(f'GBS-Clustering for graph {row["id"]}...')
      if int(row["id"])==10 or int(row["id"])==29 :
          cluster_list.append([])
          parameter_list.append([])
          pass
      else:
          Adj=row["Adj"]
          np.fill_diagonal(Adj, 0)
          clusters=GBS_Based_Clustering(N=nsamples,L=int(sim_params["n_subspace"]/3) ,n_mean=int(sim_params["n_subspace"]/2),  
                                      params_GBS_Sampler=sim_params, Adj=Adj,foldername=row["id"],tinit=n_dimension,weights=np.ones(sim_params["n_subspace"]),MinPoints=MinPoints)
          cluster_list.append(clusters)
          parameter_list.append([nsamples,int(sim_params["n_subspace"]/3),int(sim_params["n_subspace"]/2),n_dimension,MinPoints])

          print(f'GBS-Clustering for graph {row["id"]} done')
    
    cluster_results_dataframe.insert(2, "Clusters", cluster_list, True)
    cluster_results_dataframe.insert(3, "Parameters", parameter_list, True)

    # with open(cluster_results_filename+'.pkl', 'wb') as f:
    #     pickle.dump(cluster_results_dataframe, f)
    
    # cluster_results_dataframe.to_csv(cluster_results_filename+'.csv', index=False)
    
    # Load the clusters results from the pickle file
    
    with open('clusters_results.pkl', 'rb') as f:
        cluster_results_dataframe = pickle.load(f)
    # fig, axs = plt.subplots(nrows=5, ncols=4)

    # # Set labels and titles outside the loop (optional)
    # plt.xlabel('x')
    # plt.ylabel('y')

    # for counter, row in cluster_results_dataframe.iterrows():
    #   if row["id"] in ["basic4.csv", "chrome.csv", "outliers.csv", "spirals.csv"]:
    #       pass  # Skip excluded datasets
    #   else:
    #     # Customize scatter plot (optional)
    #     axs[counter // 6, counter % 4].scatter(row["x"], row["y"], color='blue', marker='o')

    #     # Extract a more descriptive title (optional)
    #     title = row["id"].split(".")[0]  # Assuming filename format
    #     axs[counter // 6, counter % 4].set_title(title)

    # fig.tight_layout()
    # plt.show()
 
    # Compute the intra and inter cluster cohesion for each dataset
    KMeans_results=[]
    DBSCAN_results=[]
    GBS_results=[]
    for index,row in cluster_results_dataframe.iterrows():
        print(f'Computing metrics for graph {row["id"]}...') # Clusters from GBS based clustering
        if int(row["id"])==10 or int(row["id"])==29 :
            KMeans_results.append([0,0])
            DBSCAN_results.append([0,0])
            GBS_results.append([0,0])
        else: 
            clusters=row["Clusters"]
            label_GBS=generate_label_fromGBSclusters(clusters)
            Adj=row["Adj"]
            np.fill_diagonal(Adj, 0)
            DistanceMatrix=row["DistanceMatrix"]
            GBS_results.append([intra_inter_cluster_cohesion(Adj, label_GBS),density_metric(Adj,label_GBS)])
            # Coordinates of the points to be clustered
            data=np.array([row["x"],row["y"]]).T
            # KMeans
            kmeans = KMeans(n_clusters=len(clusters), random_state=0,n_init='auto').fit(DistanceMatrix)
            labels_kmeans=kmeans.labels_
            KMeans_results.append([intra_inter_cluster_cohesion(Adj, labels_kmeans),density_metric(Adj,labels_kmeans)])
            # DBSCAN
            dbscan = DBSCAN(eps=0.005, min_samples=1).fit(data)
            labels_dbscan=dbscan.labels_
            DBSCAN_results.append([intra_inter_cluster_cohesion(Adj, labels_dbscan),density_metric(Adj,labels_dbscan)])

        
    

    KMeans_results=np.array(KMeans_results)
    DBSCAN_results=np.array(DBSCAN_results)
    GBS_results=np.array(GBS_results)
    cluster_results_dataframe.insert(4, "KMeans_density", KMeans_results[:,1], True)
    cluster_results_dataframe.insert(5, "DBSCAN_density", DBSCAN_results[:,1], True)
    cluster_results_dataframe.insert(6, "KMeans_intra_inter", KMeans_results[:,0], True)
    cluster_results_dataframe.insert(7, "DBSCAN_intra_inter", DBSCAN_results[:,0], True)
    cluster_results_dataframe.insert(8, "GBS_density", GBS_results[:,1], True)
    cluster_results_dataframe.insert(9, "GBS_intra_inter", GBS_results[:,0], True)
    print(f'Mean for Intra-inter cluster cohesion and density for KMeans: {[np.mean(KMeans_results[:,0]),np.mean(KMeans_results[:,1])]}')
    print(f'Mean Intra-inter cluster cohesion and density for DBSCAN: {[np.mean(DBSCAN_results[:,0]),np.mean(DBSCAN_results[:,1])]}')
    print(f'Mean Intra-inter cluster cohesion and density for GBS: {[np.mean(GBS_results[:,0]),np.mean(GBS_results[:,1])]}')
    print(f'Std for Intra-inter cluster cohesion and density for KMeans: {[np.std(KMeans_results[:,0]),np.std(KMeans_results[:,1])]}')
    print(f'Std Intra-inter cluster cohesion and density for DBSCAN: {[np.std(DBSCAN_results[:,0]),np.std(DBSCAN_results[:,1])]}')
    print(f'Std Intra-inter cluster cohesion and density for GBS: {[np.std(GBS_results[:,0]),np.std(GBS_results[:,1])]}')
    cluster_results_dataframe.to_csv('cluster_results.csv', index=False)
        

