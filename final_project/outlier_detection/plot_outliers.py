import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_outliers_each_dim(data, predictions):
    plt.figure(figsize=(15, 5))

    for i, col in enumerate(data.columns):
        x_coord_1 = np.random.normal(i + 2, 0.1, len((data[predictions == 1])[col]))
        plt.scatter(x_coord_1, (data[predictions == 1])[col], c='green', edgecolors='black', alpha=0.7)
        x_coord_neg_1 = np.random.normal(i + 2, 0.1, len((data[predictions == -1])[col]))
        plt.scatter(x_coord_neg_1, (data[predictions == -1])[col], c='red', edgecolors='black')
    plt.xticks([i + 2 for i in range(len(data.columns))],
               [column_name.replace("_", " ").title() for column_name in data.columns])
    plt.xlabel('Dimension')
    plt.ylabel('Normalized Value')
    plt.title('Outliers Plotted For Each Dimension Individually')

    plt.show()

def plot_outliers_pca_2(data, predictions):
    plt.figure(figsize=(15, 5))

    # Perform PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    data['PCA1'] = pca_result[:, 0]
    data['PCA2'] = pca_result[:, 1]

    plt.scatter(data['PCA1'], data['PCA2'], c='green', edgecolors='black', alpha=0.7)
    plt.scatter((data[predictions == -1])['PCA1'], (data[predictions == -1])['PCA2'], c='red', edgecolors='black')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title('Outliers Plotted With 2 PCA Dimensions')

    plt.show()
