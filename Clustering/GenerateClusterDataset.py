from ClusteringMethods import *
from sklearn.datasets import make_classification

if __name__ == '__main__':
   
    current_dir = os.path.dirname(__file__)
    os.chdir(current_dir)
    data_file = 'dataset\\adjacency_matrix_generated_dataset'
    n_dimension=24

    adjacency_matrix_dataframe=pd.DataFrame({'id': [], 'DistanceMatrix': [],'x': [],'y': []})
    adjacency_matrix_list=[]
    for i in range(30):
        print(f'Computing distance matrix for graph {i}...')
        n=np.random.randint(1,4)
        if n==1:
            X, _ = make_classification(n_samples=n_dimension,
    n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)
        elif n==2:
            X, _ = make_classification(n_samples=n_dimension,n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)
        elif n==3:
            X, _ = make_classification(n_samples=n_dimension,n_features=2, n_redundant=0, n_informative=2)
        elif n==4:
            X, _ = make_classification(n_samples=n_dimension,n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_classes=3)

        x=X[:,0]
        y=X[:,1]
        print(len(x))
        print(len(y))
        data=np.array([x,y]).T
        print(len(data))
        distance_matrix=np.array(computeDistanceMatrix(data))
        adjacency_matrix_list.append(buildAdjacencyMatrix(distance_matrix,np.quantile(distance_matrix, 0.35, axis=0)))
        
        adjacency_matrix_dataframe = pd.concat([adjacency_matrix_dataframe,pd.DataFrame({'id': i, 
                                                                                       'DistanceMatrix': [distance_matrix], 
                                                                                        'x':[x],
                                                                                        'y':[y],
                                                                                        })],
                                                                                          ignore_index=True)

    adjacency_matrix_dataframe.insert(1, "Adj", adjacency_matrix_list, True)
    adjacency_matrix_dataframe.to_pickle(data_file+'.pkl')
    adjacency_matrix_dataframe.to_csv(data_file+'.csv', index=False)