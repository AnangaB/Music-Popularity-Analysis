from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

def get_features(X, y, model_name="svm"):
    """
    Select features using Recursive Feature Elimination (RFE), implemented using RFECV from sklearn, with 3-fold cross-validation. Since classification later is using either SVM or Random Forest, the estimator used for RFE is either SVC or RandomForestClassifier.
    

    """
    if model_name == "svm":
        # must be linear for RFECV/RFE
        estimator = SVC(kernel="linear", random_state=42)
    elif model_name == "rf":
        estimator = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError("model must be 'svm' or 'rf'")
    
    selector = RFECV(
        estimator=estimator,
        step=1,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )
    
    selector.fit(X, y)
    selected_features = X.columns[selector.support_]
    print("Optimal number of features:", selector.n_features_)
    return selected_features, selector
