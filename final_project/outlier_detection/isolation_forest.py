from sklearn.ensemble import IsolationForest


def calculate_outliers_isolation_forest(data):
    isolation_forest = IsolationForest(random_state=67, contamination=0.01)

    predictions = isolation_forest.fit_predict(data)

    return predictions
