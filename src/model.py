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

default_training = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data", "processed", "test_heart.csv")
default_test = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data", "processed", "test_heart.csv")
default_to = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "results")

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

def main(training_path, test_path, to_path):
    model(training_path, test_path, to_path)
    
if __name__ == "__main__":  
  main(opt["--training"], opt["--test"], opt["--to"])