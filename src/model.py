# author: 
# date: 

"""
A utility script to build and fit ML model(s) ....

Usage: model.py [--from=<raw_file_path>] [--to=<processed_dir_path>]
Options:
[--from=<file path >]     Path to the directory where the pre_processed data is stored

[--to=<out_file>]         Directory path save the results of the model
                            
Uses the docopt for command-line argument parsing (add other packages used)
- http://docopt.org/
"""

from docopt import docopt
import os
import pandas as pd

opt = docopt(__doc__)

def model(from_path, to_path):
    """
    A utility script to build and fit ML model(s)

    Parameters
    ----------
    from_path : _type_
        Path to the directory where the pre_processed data is stored
    to_path : _type_
        Directory path save the results of the model
    """
    print("\nTraining the model(s): to be implemented ...")
    print(f"Reading data from {from_path}")
    print(f"Saving model artifacts to {to_path}")

def main(from_path, to_path):
    model(from_path, to_path)
 
if __name__ == "__main__":
  main(opt["--from"], opt["--to"])