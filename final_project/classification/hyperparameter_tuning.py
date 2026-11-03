from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform



def tune_random_forest_model(X_train, y_train):

    # Random Forest Classifier
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    rf_param_dist = {
        "n_estimators": randint(100, 600),
        "max_depth": randint(3, 40),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    }

    rf_random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_param_dist,
        n_iter=15,
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    rf_random_search.fit(X_train, y_train)

    print("Best RF params:", rf_random_search.best_params_)
    print("Best RF CV score:", rf_random_search.best_score_)

    best_rf = rf_random_search.best_estimator_
    return best_rf




def tune_svm_model(X_train, y_train):

    # Support Vector Machine Classifier
    svm = SVC(random_state=42)

    svm_param_dist = {
    "C": uniform(0.01, 100),
    "kernel": ["linear", "rbf"]
    }

    svm_random_search = RandomizedSearchCV(
        estimator=svm,
        param_distributions=svm_param_dist,
        n_iter=15,
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    svm_random_search.fit(X_train, y_train)

    print("Best SVM params:", svm_random_search.best_params_)
    print("Best SVM CV score:", svm_random_search.best_score_)

    best_svm = svm_random_search.best_estimator_
    return best_svm


