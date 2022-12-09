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
# make heart.csv

PROJECT_NAME := heart_disease_predictor
CONDA_ENV := env_heart_disease_prediction
CWD := $(shell pwd)
RESULTS := $(CWD)/_output/results
RAW := $(CWD)/_output/data/raw
PROCESSED := $(CWD)/_output/data/processed

all: init _output/book.pdf

# download data
_output/data/raw/heart.csv : src/fetch_dataset.py
	@echo "downloading data from UCI Machine Learning Repository"
	@python src/fetch_dataset.py --to=_output/data/raw/heart.csv

# pre-process data (i.e., change column names and split the data into train and test sets)
_output/data/processed/test_heart.csv _output/data/processed/train_heart.csv : src/pre_process.py _output/data/raw/heart.csv
	@echo "processing the downloaded data"
	@python src/pre_process.py --from=_output/data/raw/heart.csv --to=_output/data/processed

# exploratory data analysis (i.e., visualize feature and target distributions)
_output/results/categorical_distributions.png _output/results/correlation_scatter.png  _output/results/thalach_vs_age.png _output/results/numeric_distributions.png _output/results/correlation_matrix.png _output/results/class_count.csv: src/eda.py  _output/data/processed/train_heart.csv  
	@echo "performing exploratory data analysis"
	@python src/eda.py --from=_output/data/processed --to=_output/results

# model selection and training (comparison of SVC and logistic reg model performance using f1 scoring)
_output/results/model_selection_results.csv  _output/results/optimization_results.csv _output/results/confusion_matrix.png: src/model.py _output/results/categorical_distributions.png _output/results/correlation_scatter.png  _output/results/thalach_vs_age.png  _output/results/numeric_distributions.png _output/results/correlation_matrix.png _output/results/class_count.csv
	@echo "training the model"
	@python src/model.py --training=_output/data/processed/train_heart.csv  --test=_output/data/processed/test_heart.csv --to=_output/results

# render report
_output/book.pdf : doc/heart_disease_prediction_report/_toc.yml _output/results/model_selection_results.csv  _output/results/optimization_results.csv _output/results/confusion_matrix.png  _output/results/correlation_scatter.png  _output/results/thalach_vs_age.png _output/results/categorical_distributions.png _output/results/numeric_distributions.png _output/results/correlation_matrix.png _output/results/class_count.csv 
	@echo "generating the pdf report"
	@cd doc/heart_disease_prediction_report && jupyter-book build . --builder pdfhtml
	@cp doc/heart_disease_prediction_report/_build/pdf/book.pdf _output/book.pdf

clean:
	@rm -rf $(CWD)/_output
	@rm -rf $(CWD)/doc/heart_disease_prediction_report/_build

init:
	@mkdir -p $(CWD)/_output
	@mkdir -p $(CWD)/_output/data
	@mkdir -p $(PROCESSED)
	@mkdir -p $(RAW)
	@mkdir -p $(RESULTS)
