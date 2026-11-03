from sklearn.neighbors import LocalOutlierFactor

def calculate_outliers_local_outlier_factor(data):
    local_outlier_factor = LocalOutlierFactor(n_neighbors=20)

    predictions = local_outlier_factor.fit_predict(data)

    return predictions
