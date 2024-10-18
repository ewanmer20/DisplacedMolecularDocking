from DGBS_TaceAs_classes import *
from sklearn.cluster import KMeans,DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances

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
    return np.array([[1 if D[i][j]<d_tilda else 0 for j in range(len(D))] for i in range(len(D))])

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
        if Density(subgraph)>=dbest:
            if Density(subgraph)==dbest:
                if np.sum(samples[i])>np.sum(best):
                    dbest=Density(subgraph)
                    best=samples[i]
            else:
                dbest=Density(subgraph)
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
    remaining_nodes=remaining_nodes-best
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
            if np.sum(reduced_subgraph[i,:])==0:
                subgraph=np.zeros(len(remaining_nodes))
                subgraph[i]=1
                clusters.append(subgraph)
                remaining_nodes[i]=0
            else:
                ratio_connectivity=np.zeros(len(clusters))
                for i in range(clusters):
                    reduced_subgraph=reduce_adjacency_matrix(Adj,clusters[i])
                    ratio_connectivity[i]=np.sum([Adj[i,j]+Adj[j,i] for j in range(len(Adj)) if clusters[i][j]>0])/(2*np.sum(clusters[i]))
                index_cluster=np.argmax(ratio_connectivity)
                clusters[index_cluster][i]=1
    return clusters

def GBS_Based_Clustering(data, N, L, n_mean, d_tilda,params_GBS_Sampler,D):
    D=computeDistanceMatrix(data)
    Adj=buildAdjacencyMatrix(D,d_tilda)
    remaining_nodes=np.ones(len(Adj))
    clusters=[]
    params_GBS_Sampler_copy=copy.deepcopy(params_GBS_Sampler)
    params_GBS_Sampler_copy["Adj"]=Adj
    params_GBS_Sampler_copy["n_subspace"]= N
    params_GBS_Sampler_copy["target_nsqz"]= n_mean
    DGBS_Sampler=DGBS_Sampler(**params_GBS_Sampler)
    while BigEnough(Adj):
        i=0
        Go=True
        while Go:
            samples=DGBS_Sampler.run_sampler(nsamples=N,foldername="test")
            samples=postSelectSamples(samples,L)
            best, dbest= findDensestCandidate(samples,Adj)
            t=computeThreshold(t,i)
            if dbest>t:
                clusters.append(best)
                Go=False
            i+=1
        remaining_nodes=removeFoundCluster(remaining_nodes,best)
        L, n_mean=updateParameters(Adj)
    return postProcessing(clusters,remaining_nodes,Adj)


def weighted_density(data,clusters):
    """
    Compute the weighted density
    :param data: the data
    :param clusters: the clusters
    :return: the weighted density
    """
    return None
def intra_inter_cluster_cohesion(data,clusters):
    """
    Compute the intra and inter cluster cohesion
    :param data: the data
    :param clusters: the clusters
    :return: the intra and inter cluster cohesion
    """
    # Calculate intra-cluster cohesion
    intra_cohesion = 0
    for cluster_label in set(labels):
        cluster_data = data[labels == cluster_label]
    intra_cohesion += pairwise_distances(cluster_data).mean() / len(cluster_data)

    # Calculate inter-cluster cohesion
    inter_cohesion = 0
    for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        if labels[i] != labels[j]:
            inter_cohesion += np.linalg.norm(data[i] - data[j])
    inter_cohesion /= (len(labels) * (len(labels) - 1)) / 2

if __name__ == '__main__':
    #TODO: write tests for the functions
  adjacency_matrix = np.array([[0, 1, 0, 1],
                               [1, 0, 1, 0],
                               [0, 1, 0, 1],
                               [1, 0, 1, 0]])

  # Select nodes 0 and 2
  selected_nodes = np.array([1, 0, 1, 0])

  # Get the subgraph adjacency matrix
  subgraph_adjacency_matrix = reduce_adjacency_matrix(adjacency_matrix, selected_nodes)

  # Print the subgraph adjacency matrix
  print(subgraph_adjacency_matrix)
    