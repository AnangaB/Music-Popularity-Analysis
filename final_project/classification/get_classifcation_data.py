from sklearn.model_selection import train_test_split


def get_train_and_test_data(data, target, optimal_features=None):
    """
    Get the training and testing data for classification tasks. If optimal_features is provided, only those features will be used.
    Returns:
        X_train, X_test, y_train, y_test: Split datasets for training and testing.
    """

    if optimal_features is not None:
        data = data[optimal_features]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=42, stratify=target
    )

    return X_train, X_test, y_train, y_test