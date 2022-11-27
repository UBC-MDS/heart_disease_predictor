# author: Tony Zoght
# date: 2022-11-18

"""
A utility script to download the dataset used in (https://github.com/UBC-MDS/heart_disease_predictor) 
If no arguments are provided, the script will download the dataset from the default url to the default path
default url: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
default path: data/raw/heart.csv

Usage: fetch_dataset.py [--from=<url>] [--to=<out_file>]
Options:
[--from=<url>]             URL from where to download the dataset (must be in standard csv format)
                           If not provided, the default url will be used 
[--to=<out_file>]          Path (including filename) of where to locally write the file
                           If not provided, the default path will be used 
    
    
                           
Uses the docopt for command-line argument parsing and pandas to read the and save the dataset
- http://docopt.org/
- https://pandas.pydata.org/
"""

from docopt import docopt
import os
import pandas as pd

default_from = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
default_to = os.path.join(os.path.dirname(__file__), os.pardir, "data", "raw", "heart.csv")

opt = docopt(__doc__)

def fetch_dataset(from_url, to_file_path):
    """
    This function will download the dataset from the default url to the default path

    Parameters
    ----------
    from_url : str
        URL from where to download the dataset (must be in standard csv format)
        If not provided, the default url will be used 
    to_file_path : str
        Path (including filename) of where to locally write the file
        If not provided, the default path will be used 
    """
    if (from_url is None):
        from_url = default_from
    if (to_file_path is None):
        to_file_path = default_to
    print(f"Downloading dataset from {from_url} to {to_file_path}")
    data = pd.read_csv(from_url, header=None)
    if not os.path.exists(to_file_path):
        os.makedirs(to_file_path)
    print(f"Saving dataset to {to_file_path}")
    data.to_csv(to_file_path, index = False, header=False)

def main(from_url, to_path):
    fetch_dataset(from_url, to_path)
 
if __name__ == "__main__":
  main(opt["--from"], opt["--to"])
