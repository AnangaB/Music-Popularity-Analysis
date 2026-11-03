from sklearn.cluster import KMeans
from visualize_cluster import reduce_visualize_and_print_score


def kmeans_cluster(data):
    kmeans = KMeans(n_clusters=4, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data)

    reduce_visualize_and_print_score(data, "KMeans Clustering")

