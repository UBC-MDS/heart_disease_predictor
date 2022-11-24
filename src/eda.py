# author: 
# date: 

"""
A utility script to perform exploratory data analysis on the dataset used in ....

Usage: eda.py [--from=<raw_file_path>] [--to=<processed_dir_path>]
Options:
[--from=<file path >]     Path to the directory where the pre_processed data is stored

[--to=<out_file>]          Directory path to save the EDA artifacts
                            
Uses the docopt for command-line argument parsing (add other packages used)
- http://docopt.org/
"""

from docopt import docopt
import os
import pandas as pd

opt = docopt(__doc__)

def eda(from_path, to_path):
    """
    This function will perform exploratory data analysis on the dataset and save the artifacts in the specified directory

    Parameters
    ----------
    from_path : str
        Path to the directory where the pre_processed data is stored
    to_path : str
        Directory path to save the EDA artifacts
    """
    print("\nPerforming EDA on the dataset: to be implemented ...")
    print(f"Reading data from {from_path}")
    print(f"Saving EDA artifacts to {to_path}")

def main(from_path, to_path):
    eda(from_path, to_path)
 
if __name__ == "__main__":
  main(opt["--from"], opt["--to"])