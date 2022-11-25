# author: 
# date: 

"""
A utility script to build and fit ML model(s) ....

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
from sklearn.metrics import ConfusionMatrixDisplay


default_training = os.path.join(os.path.dirname(__file__), os.pardir, "data", "processed", "train_heart.csv")
default_test = os.path.join(os.path.dirname(__file__), os.pardir, "data", "processed", "test_heart.csv")
default_to = os.path.join(os.path.dirname(__file__), os.pardir, "results")

opt = docopt(__doc__)

def model(training_path, test_path, to_path):
    if (training_path is None):
        training_path = default_training
    if (test_path is None):
        test_path = default_test
    if (to_path is None):
        to_path = default_to
    print("\nTraining the model(s): to be implemented ...")
    print(f"Reading training data from {training_path}")
    print(f"Reading test data from {test_path}")
    print(f"Saving model artifacts to {to_path}")
    
    train_df = pd.read_csv(training_path)
    test_df = pd.read_csv(test_path)

    # age,sex,chest_pain_type,resting_blood_pressure,cholesterol,
    # fasting_blood_sugar,resting_ecg_results,max_hr_achieved,
    # exercise_induced_angina,oldpeak,slope,num_major_vessels,
    # thalassemia,target

    numeric_features = ["age", "resting_blood_pressure", "cholesterol", "max_hr_achieved", "oldpeak"]
    passthrough_features = ["sex", "chest_pain_type", "fasting_blood_sugar", "resting_ecg_results", "exercise_induced_angina", "slope", "num_major_vessels", "thalassemia"]

    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        ("passthrough", passthrough_features)
    )

    X_train, y_train = train_df.drop(columns=["target"]), train_df["target"]
    X_test, y_test = test_df.drop(columns=["target"]), test_df["target"]

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

    results["dummy"] = pd.DataFrame(cross_validate(dc, X_train, y_train, cv=5, return_train_score=True, scoring="f1")).agg(['mean','std']).round(3).T
    results["SVM"] = pd.DataFrame(cross_validate(pipe_svm, X_train, y_train, cv=5, return_train_score=True, scoring="f1")).agg(['mean','std']).round(3).T
    results["log_reg"] = pd.DataFrame(cross_validate(pipe_lr, X_train, y_train, cv=5, return_train_score=True, scoring="f1")).agg(['mean','std']).round(3).T

    results_df = pd.concat(results, axis='columns')
    results_df.to_csv(os.path.join(to_path,"model_selection_results.csv"),index = True, header=results_df.columns)

    param_dist ={
        "logisticregression__C": loguniform(1e-3, 1e3) 
    }

    random_search = RandomizedSearchCV(pipe_lr, param_distributions=param_dist, n_jobs=-1, n_iter=20, cv=5, random_state=123, refit="f1", scoring=["f1", "recall", "precision"], return_train_score=True)
    random_search.fit(X_train, y_train)
    optim_results_df = pd.DataFrame(random_search.cv_results_)[
        [
            "mean_fit_time",
            "mean_score_time",
            "param_logisticregression__C",
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

    optim_results_df.to_csv(os.path.join(to_path,"optimization_results.csv"),index = True, header=optim_results_df.columns)
    
    test_f1_score = random_search.score(X_test, y_test)
    print(f"The test f1 score was {test_f1_score}")

    cm = ConfusionMatrixDisplay.from_estimator(
    random_search, X_test, y_test, values_format="d", display_labels = ["No Heart Disease", "Heart Disease"]
    )

    cm.to_png(os.path.join(to_path,"confusion_matrix.png"))

def main(training_path, test_path, to_path):
    model(training_path, test_path, to_path)
    
if __name__ == "__main__":  
  main(opt["--training"], opt["--test"], opt["--to"])