import pandas as pd


def print_outlier_amounts(predictions):
    predictions_list = predictions.tolist()
    num_inliers = sum([1 for val in predictions_list if val == 1])
    num_outliers = sum([1 for val in predictions_list if val == -1])

    print(f"Inlier amount: {num_inliers}, Outlier amount: {num_outliers}")

def print_outliers(data, predictions):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print("The Outliers are:")
    print(data[predictions == -1])
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
