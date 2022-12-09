# author: 
# date: 

"""
A utility script to build and fit ML model(s)

Usage: model.py [--training=<path to training data set>] [--test=<path to test data set >] [--to=<out_file>]

Options:
[--training=<file path to training data set>]     Path to the training dataset csv file
[--test=<file path to test data set >]             Path to the test dataset csv file
[--to=<out_file>]                                  Where to save the model artifacts
"""

from docopt import docopt
import os
import pandas as pd
from scipy.stats import loguniform
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, f1_score


default_training = os.path.join(os.path.dirname(__file__), os.pardir, "data", "processed", "train_heart.csv")
default_test = os.path.join(os.path.dirname(__file__), os.pardir, "data", "processed", "test_heart.csv")
default_to = os.path.join(os.path.dirname(__file__), os.pardir, "results")

opt = docopt(__doc__)

# Attributed to Varada, DSCI 571
def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)

def model(training_path, to_path):
    """
    Function that first preprocesses the data, then conducts model selection between SVC and logistic regression, 
    and finally conducts hyperparmeter optimization using RandomSearch.

    Parameters
    ----------
    training_path : str
        Path to training data [default: default_training]
    to_path : str
        Path to save the training results [default: default_to]

    Returns
    ----------
    random_search: sklearn estimator
        Fitted model with best parameters based on top f1 score using RandomSearch
    """
    if (training_path is None):
        training_path = default_training
    if (to_path is None):
        to_path = default_to
    print("\nTraining the model(s):")
    print(f"Reading training data from {training_path}")
    
    train_df = pd.read_csv(training_path)

    numeric_features = ["age", "resting_blood_pressure", "cholesterol", "max_hr_achieved", "oldpeak"]
    passthrough_features = ["sex", "chest_pain_type", "fasting_blood_sugar", "resting_ecg_results", "exercise_induced_angina", "slope", "num_major_vessels", "thalassemia"]

    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        ("passthrough", passthrough_features)
    )

    X_train, y_train = train_df.drop(columns=["target"]), train_df["target"]

    pipe_lr = make_pipeline(
        preprocessor,
        LogisticRegression()
    )
    pipe_svm = make_pipeline(
        preprocessor,
        SVC()
    )

    dc = DummyClassifier()
    results = {}

    results["dummy"] = mean_std_cross_val_scores(dc, X_train, y_train, cv=5, return_train_score=True, scoring="f1")
    results["SVM"] = mean_std_cross_val_scores(pipe_svm, X_train, y_train, cv=5, return_train_score=True, scoring="f1")
    results["log_reg"] = mean_std_cross_val_scores(pipe_lr, X_train, y_train, cv=5, return_train_score=True, scoring="f1")

    results_df = pd.concat(results, axis='columns')
    print(f"Saving model artifacts to {to_path}")
    results_df.to_csv(os.path.join(to_path,"model_selection_results.csv"),index = True, header=results_df.columns)

    param_dist ={
        "svc__C": loguniform(1e-3, 1e3),
        "svc__gamma": loguniform(1e-3, 1e3)  
    }

    print(f"Finding optimal hyperparameters")
    random_search = RandomizedSearchCV(pipe_svm, param_distributions=param_dist, n_jobs=-1, n_iter=50, cv=5, random_state=123, refit="f1", scoring=["f1", "recall", "precision"], return_train_score=True)
    random_search.fit(X_train, y_train)
    optim_results_df = pd.DataFrame(random_search.cv_results_)[
        [
            "mean_fit_time",
            "mean_score_time",
            "param_svc__C",
            "param_svc__gamma",
            "mean_test_f1",
            "std_test_f1",
            "rank_test_f1",
            "mean_train_f1",
            "std_train_f1",
            "mean_test_recall",
            "std_test_recall",
            "rank_test_recall",
            "mean_train_recall",
            "std_train_recall",
            "mean_test_precision",
            "std_test_precision",
            "rank_test_precision",
            "mean_train_precision",
            "std_train_precision"   
        ]
    ].set_index("rank_test_f1").sort_index().T

    print(f"The best parameters were {random_search.best_params_}")
    
    optim_results_df.to_csv(os.path.join(to_path,"optimization_results.csv"),index = True, header=optim_results_df.columns)
    
    return random_search

def test(test_path, model, to_path):
    """
    Test function that scores a fitted model on the test data using
    the f1 metric.

    Parameters
    ----------
    test_path : str
        Path to test data [default: default_test]
    model : sklearn estimator
        Fitted model to test
    to_path : str
        Path to save the training results [default: default_to]
    """
    if (test_path is None):
        test_path = default_test
    if (to_path is None):
        to_path = default_to
    print(f"Reading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    X_test, y_test = test_df.drop(columns=["target"]), test_df["target"]

    test_predictions = model.predict(X_test)
    test_f1_score = f1_score(y_test, test_predictions)
    print(f"The test f1 score was {test_f1_score}")

    cm = ConfusionMatrixDisplay.from_estimator(
    model, X_test, y_test, values_format="d", display_labels = ["No Heart Disease", "Heart Disease"]
    )

    cm.figure_.savefig(os.path.join(to_path,"confusion_matrix.png"))

def main(training_path, test_path, to_path):
    """
    Driver function that applies machine learning models on the data
    and saves the artifacts that are generated.

    Parameters
    ----------
    training_path : str
        Path to the training data [default: default_training]
    test_path : str
        Path to test data [default: default_test]
    to_path : str
        Path to save the training results [default: default_to]
    """
    my_model = model(training_path, to_path)
    test(test_path, my_model, to_path)
    
if __name__ == "__main__":  
  main(opt["--training"], opt["--test"], opt["--to"])