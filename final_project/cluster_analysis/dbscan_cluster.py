from sklearn.cluster import DBSCAN
from visualize_cluster import reduce_visualize_and_print_score


def dbscans_cluster(data):

    db = DBSCAN(eps=0.2, min_samples=10)
    labels = db.fit_predict(data)
    
    data["Cluster"] = labels

    print("DBSCAN Clusters:")
    print(data["Cluster"].value_counts())

    reduce_visualize_and_print_score(data, "DBSCAN Clustering")

