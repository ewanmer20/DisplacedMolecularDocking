from ClusteringMethods import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import numpy as np

if __name__=="__main__":

    # Load the adjacency matrix and the distance matrix
    current_dir = os.path.dirname(__file__)
    os.chdir(current_dir)
    filename_adj_df='dataset\\adjacency_matrix_generated_dataset'
    with open('clusters_results.pkl', 'rb') as f:
        cluster_results_dataframe = pickle.load(f)


    nsamples=1000
    n_dimension=24
    MinPoints=3
    graph_index=0 # Index of the cluster datastet to be analayzed 
    sqz_values=np.linspace(2,12,10) # Values of squeezing to be tested
    displacement_values=np.linspace(0,12,10) # Values of displacement to be tested


    result=pd.DataFrame({'id':[],'sqz':[],'displacement':[],'Adj':[],'GBS_density':[],'Kmeans density':[]})


    Adj=cluster_results_dataframe.iloc[graph_index]["Adj"]
    np.fill_diagonal(Adj, 0)
    foldername=cluster_results_dataframe.iloc[graph_index]["id"]
    DistanceMatrix=cluster_results_dataframe.iloc[graph_index]["DistanceMatrix"]
    print(f'Graph id is:{foldername}')



    for i in range(len(sqz_values)):
        for j in range(len(displacement_values)):



            sim_params = {
        "tau": 1.1,
        "alpha": 2.1,
        "target_nsqz": sqz_values[i],
        "target_ncoh": displacement_values[j],
        "loss_mode": 0.0,
        "hbar": 2,
        "n_subspace": n_dimension,
        "conv": "real",
        "save": False,
        "arb_graph": True
                        }
         
            # clusters=GBS_Based_Clustering(N=nsamples,L=int(sim_params["n_subspace"]/3) ,n_mean=sqz_values[i],  
            #                     params_GBS_Sampler=sim_params, Adj=Adj,foldername=foldername,tinit=n_dimension,weights=np.ones(sim_params["n_subspace"]),MinPoints=MinPoints)
            clusters=GBS_Based_Clustering_Alternative(N=nsamples,L=int(sim_params["n_subspace"]/3) ,n_mean=sqz_values[i],  
                                params_GBS_Sampler=sim_params, Adj=Adj,foldername=foldername,weights=np.ones(sim_params["n_subspace"]),MinPoints=MinPoints)
            if clusters==[]:
                print(f'No clusters found for i={i} j={j}')
                continue
    
            label_GBS=generate_label_fromGBSclusters(clusters)
            kmeans = KMeans(n_clusters=len(clusters), random_state=0,n_init='auto').fit(DistanceMatrix)
            labels_kmeans=kmeans.labels_
            result = pd.concat([result,pd.DataFrame({'id': foldername, 'sqz': sqz_values[i], 'displacement': displacement_values[j],
                                                      'Adj': [Adj], 'GBS_density': density_metric(Adj,label_GBS), 
                                                      'Kmeans density': density_metric(Adj,labels_kmeans)})], ignore_index=True)
            print(f'Iteration i={i} j={j} completed')
            
    result.to_pickle(f'results_varying_displacement_and_squeezing_alternative_clustering{foldername}.pkl')

    # Plot the results

    # Create grid values
    sqz = np.linspace(result['sqz'].min(), result['sqz'].max(), 100)
    disp = np.linspace(result['displacement'].min(), result['displacement'].max(), 100)
    sqz, disp = np.meshgrid(sqz, disp)

    # Interpolate GBS_density values on the grid
    GBS_density = griddata((result['sqz'], result['displacement']), result['GBS_density'], (sqz, disp), method='cubic')

     
    Kmeans_density = griddata((result['sqz'], result['displacement']), result['Kmeans density'], (sqz, disp), method='cubic')

    # Create a surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(sqz, disp, GBS_density, cmap='viridis')

    # Set labels
    ax.set_xlabel('Squeezing (sqz)')
    ax.set_ylabel('Displacement (disp)')
    ax.set_zlabel('GBS Density')

    # Set title
    ax.set_title('GBS Density as a function of Squeezing and Displacement')

    # Add a color bar which maps values to colors
    fig.colorbar(surf)

    # Save the figure as an SVG file
    fig.savefig('GBS_density_plot_alternative_clustering.svg', format='svg')
     # Create a surface plot
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')

    # Plot the surface
    surf2 = ax2.plot_surface(sqz, disp, Kmeans_density, cmap='viridis')

    # Set labels
    ax2.set_xlabel('Squeezing (sqz)')
    ax2.set_ylabel('Displacement (disp)')
    ax2.set_zlabel('Kmeans Density')

    # Set title
    ax2.set_title('Kmeans Density as a function of Squeezing and Displacement')

    # Add a color bar which maps values to colors
    fig2.colorbar(surf2)

    fig2.savefig('Kmeans_density_plot_test_alternative_clustering.svg', format='svg')

    # Show plot
    plt.show()

                                                                                        