# author: 
# date: 

"""
A utility script to perform exploratory data analysis on the dataset used in ....

Usage: eda.py [--from=<raw_file_path>] [--to=<processed_dir_path>]
Options:
[--from=<path to training data >]     Path to the training dataset csv file

[--to=<directory to save results>]    Directory path to save the EDA artifacts
                            
Uses the docopt for command-line argument parsing (add other packages used)
- http://docopt.org/
"""

from docopt import docopt
import os
import pandas as pd

opt = docopt(__doc__)

default_from = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data", "processed", "train_heart.csv")
default_to = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "results")

def eda(from_path, to_path):
    if from_path is None:
        from_path = default_from
    if to_path is None:
        to_path = default_to
    print("\nPerforming EDA on the dataset: to be implemented ...")
    print(f"Reading data from {from_path}")
    print(f"Saving EDA artifacts to {to_path}")

def main(from_path, to_path):
    eda(from_path, to_path)
 
if __name__ == "__main__":
  main(opt["--from"], opt["--to"])