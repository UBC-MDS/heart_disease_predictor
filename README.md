
## Diagnosis of Heart Disease

---
Contributors: 
- [Natalie Cho](https://github.com/Natalie-cho)
- [Yurui Feng](https://github.com/Yurui-Feng)
- [Elena Ganacheva](https://github.com/elenagan)
- [Tony Zoght](https://github.com/tzoght)

For this projec we will be using the [Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci) dataset from UC Irvine Machine Learning Repository to answer the question: given common early signs from  chest pain to resting ECG, can we predict the presence of heart disease  ?

Answering this such question will help in the early detection of heart disease, which is the leading cause of death in the world [ref-1](https://www.cdc.gov/nchs/fastats/leading-causes-of-death.htm). 


The dataset contains 303 rows and 14 columns. The dataset contains 13 features and 1 target variable. The target variable is the diagnosis of heart disease (angiographic disease status) and the value 0 is for no diagnosis of heart disease and the value 1 is for the diagnosis of heart disease. The 13 features are as follows:
- age
- sex
- chest pain type (4 values)
- resting blood pressure
- serum cholestoral in mg/dl
- fasting blood sugar > 120 mg/dl
- resting electrocardiographic results (values 0,1,2)
- maximum heart rate achieved
- exercise induced angina
- oldpeak = ST depression induced by exercise relative to rest
- the slope of the peak exercise ST segment
- number of major vessels (0-3) colored by flourosopy
- thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
- The names and social security numbers of the patients were recently removed from the database, replaced with dummy values.


To answer the predictive question posed above, we will be explore using different classifiers to predict the presence of heart disease, such as Logistic Regression, K-Nearest Neighbors, Decision Tree, and Support Vector Machine. We will also be using the following metrics to evaluate the performance of the models: accuracy, precision, recall, and F1 score.

Before jumping into finding the best model and hyperparameters, we will be exploring the data to see if there are any missing values, outliers, and if there are any correlations between the features. We will also be using the following visualizations to explore the data: bar charts, histograms, and scatter plots.
All the reports, conclusions and visualizations will be available from this github repository, in the form of Jupyter Notebooks, and hosted html files.