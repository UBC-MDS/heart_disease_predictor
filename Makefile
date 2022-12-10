# Heart Disease Predictor Data Pipe
# Author: DSCI 522 Group 17
# Date: 2022-12-01

# This makefile script retrieves, preprocess heart disease dataset,
# and generate a final report including data visualization and machine
# learning analysis. The main purpose of the project is predicting 
# presence of heart disease from physiological indicators. This script 
# takes no arguments. 

# example usage:
# make all

PROJECT_NAME := heart_disease_predictor
CONDA_ENV := env_heart_disease_prediction
CWD := $(shell pwd)
RESULTS := $(CWD)/results
RAW := $(CWD)/data/raw
PROCESSED := $(CWD)/data/processed

all: docs/index.html

# download data
data/raw/heart.csv : src/fetch_dataset.py
	@echo "downloading data from UCI Machine Learning Repository"
	@python src/fetch_dataset.py --to=data/raw/heart.csv

# pre-process data (i.e., change column names and split the data into train and test sets)
data/processed/test_heart.csv data/processed/train_heart.csv : src/pre_process.py data/raw/heart.csv
	@echo "processing the downloaded data"
	@python src/pre_process.py --from=data/raw/heart.csv --to=data/processed

# exploratory data analysis (i.e., visualize feature and target distributions)
results/categorical_distributions.png results/correlation_scatter.png  results/thalach_vs_age.png results/numeric_distributions.png results/correlation_matrix.png results/class_count.csv: src/eda.py  data/processed/train_heart.csv  
	@echo "performing exploratory data analysis"
	@python src/eda.py --from=data/processed --to=results

# model selection and training (comparison of SVC and logistic reg model performance using f1 scoring)
results/model_selection_results.csv  results/optimization_results.csv results/confusion_matrix.png: src/model.py results/categorical_distributions.png results/correlation_scatter.png  results/thalach_vs_age.png  results/numeric_distributions.png results/correlation_matrix.png results/class_count.csv
	@echo "training the model"
	@python src/model.py --training=data/processed/train_heart.csv  --test=data/processed/test_heart.csv --to=results

# generate the html report
docs/index.html : doc/heart_disease_prediction_report/_toc.yml results/model_selection_results.csv  results/optimization_results.csv results/confusion_matrix.png  results/correlation_scatter.png  results/thalach_vs_age.png results/categorical_distributions.png results/numeric_distributions.png results/correlation_matrix.png results/class_count.csv 
	@echo "generating the html report in docs/ so it can be hosted on GitHub Pages"
	@cd doc/heart_disease_prediction_report && jupyter-book build . --builder html
	@cp -r doc/heart_disease_prediction_report/_build/html/* docs
	@touch docs/.nojekyll

clean:
	touch src/fetch_dataset.py
	touch data/processed/test_heart.csv && touch data/processed/train_heart.csv
	touch results/correlation_scatter.png
	touch results/model_selection_results.csv
	touch docs/index.html
	rm -rf $(CWD)/doc/heart_disease_prediction_report/_build

init:
	@mkdir -p $(PROCESSED)
	@mkdir -p $(RAW)
	@mkdir -p $(RESULTS)
	@mkdir -p docs
