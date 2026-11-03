from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np

def reduce_visualize_and_print_score(data, plot_title="Clustering Visualization"):

    # Perform PCA to reduce to 2 dimensions
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data)
    data['PCA1'] = pca_result[:, 0]
    data['PCA2'] = pca_result[:, 1]

    
    plt.figure(figsize=(8,6))

    clusters = np.unique(data['Cluster'])

    for c in clusters:
        # one artist per cluster → legend works
        subset = data[data['Cluster'] == c]
        plt.scatter(
            subset['PCA1'],
            subset['PCA2'],
            s=20,
            label=f"Cluster {c}"
        )
    
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.title(plot_title)
    plt.legend(title="Legend")
    plt.show()

    # Silhouette Score
    score = silhouette_score(data, data['Cluster'])
    print(f'Silhouette Score: {score}')
