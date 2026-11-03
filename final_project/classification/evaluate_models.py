from sklearn.metrics import RocCurveDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

def cv_with_metrics(model, X, y, cv=5):
    # Perform cross-validation and keep evaluation metrics from each fold, and return average metrics

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    fold_results = []
    for train_id, test_id in kf.split(X, y):
        X_train, X_test = X.iloc[train_id], X.iloc[test_id]
        y_train, y_test = y.iloc[train_id], y.iloc[test_id]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        preds_and_metrics = get_model_evaluation_metrics(y_test, y_pred)

        fold_results.append(preds_and_metrics)

    # average metrics across folds
    avg_metrics = pd.DataFrame(fold_results).mean().to_dict()
    
    return avg_metrics

def get_model_evaluation_metrics(y_actual, y_pred):
    """ Compute evaluation metrics for classification models.
    """
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_actual, y_pred)
    metrics['precision'] = precision_score(y_actual, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_actual, y_pred, average='weighted', zero_division=0)
    metrics['f1_score'] = f1_score(y_actual, y_pred, average='weighted', zero_division=0)
    y_actual_binary = np.vectorize(lambda val: 0 if val == "low" else 1)(y_actual)
    y_pred_binary = np.vectorize(lambda val: 0 if val == "low" else 1)(y_pred)
    metrics['roc_auc_score'] = roc_auc_score(y_actual_binary, y_pred_binary, average='weighted', labels=["low", "high"])

    return metrics

def visualize_model_with_classification_report(y_actual, y_pred,plot_title="Classification Report"):
    report = classification_report(y_actual, y_pred, zero_division=0)
    print("Classification Report:\n", report)

    #make a heatmap of the classification report
    ConfusionMatrixDisplay.from_predictions(y_actual, y_pred, cmap=plt.cm.Blues)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")    
    plt.title(plot_title)
    plt.show()


def visualize_model_with_roc_score(y_actual, y_pred, plot_title="ROC Curve"):
    y_actual_binary = np.vectorize(lambda val: 0 if val == "low" else 1)(y_actual)
    y_pred_binary = np.vectorize(lambda val: 0 if val == "low" else 1)(y_pred)
    RocCurveDisplay.from_predictions(y_actual_binary, y_pred_binary)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(plot_title)
    plt.show()