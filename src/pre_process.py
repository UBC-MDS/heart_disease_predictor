# author: Tony Zoght
# date: 2022-11-24

# Uses the docopt for command-line argument parsing and pandas to read the and save the dataset
# - http://docopt.org/

"""
A utility script to clean and pre-process the dataset used in 
the heart disease predictor project. 

This script will read the dataset from the default raw data path(data/raw), 
rename the columns,to be more readable, and particiton the dataset into a 
training and test set, before furhter analysis and model building.

The script will save the training and test set to the default processed 
data path (data/processed)

Usage: pre_process.py [--from=<raw_file_path>] [--to=<processed_dir_path>]
Options:
[--from=<url>]             Raw file path to process (must be in standard csv format)
                           If not provided, a default file name and path will be used 
[--to=<out_file>]          Directory path to save the preprocessed dataset
                           If not provided, a default path will be used 
"""

from docopt import docopt
import os
import pandas as pd
from sklearn.model_selection import train_test_split  

default_from = os.path.join(os.path.dirname(__file__), os.pardir, "data", "raw", "heart.csv")
default_to = os.path.join(os.path.dirname(__file__), os.pardir, "data", "processed")      

opt = docopt(__doc__)

column_names_dict = {
    'cp': 'chest_pain_type',
    'trestbps': 'resting_blood_pressure',
    'chol': 'cholesterol',
    'fbs': 'fasting_blood_sugar',
    'restecg': 'resting_ecg_results',
    'thalach': 'max_hr_achieved',
    'exang': 'exercise_induced_angina',
    'ca': 'num_major_vessels',
    'thal': 'thalassemia'
}

def pre_process(from_file_path, to_dir_path):
    """
    This function will read the dataset from the default raw data path(data/raw),

    Parameters
    ----------
    from_file_path : str
        Raw file path to process (must be in standard csv format)
        If not provided, a default file name and path will be used 
    to_dir_path : str
        Directory path to save the preprocessed dataset
        If not provided, a default path will be used
    """
    if (from_file_path is None):
        from_file_path = default_from
    if (to_dir_path is None):
        to_dir_path = default_to
        
    print(f"\nPre-processing dataset:\nfrom {from_file_path} \nto {to_dir_path}\n...\n")
    df = pd.read_csv(from_file_path)
    
    if not os.path.exists(to_dir_path):
        os.makedirs(to_dir_path)
    
    # renaming the columns to make them more readable
    df = df.rename(columns=column_names_dict)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=17)
    # print(df.head())
    # print(train_df.head())
    # print(test_df.head())
    
    # saving to disk
    train_df.to_csv(os.path.join(to_dir_path,"train_heart.csv"), index = False, header=df.columns)
    test_df.to_csv(os.path.join(to_dir_path,"test_heart.csv"),index = False, header=df.columns)

def main(from_file_path, to_dir_path):
    pre_process(from_file_path, to_dir_path)

if __name__ == "__main__":
  main(opt["--from"], opt["--to"])
  
