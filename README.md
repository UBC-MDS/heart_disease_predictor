## Diagnosis of Heart Disease

---
### Contributors and Maintainers
- [Natalie Cho](https://github.com/Natalie-cho)
- [Yurui Feng](https://github.com/Yurui-Feng)
- [Elena Ganacheva](https://github.com/elenagan)
- [Tony Zoght](https://github.com/tzoght)


### Proposal

For this project, we will be using the [Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci) dataset from the UC Irvine Machine Learning Repository to answer the question: given common early signs, from chest pain to resting ECG, can we predict the presence of heart disease?

Answering this question will help in the early detection of heart disease, which is the leading cause of death in the world [ref-1](https://www.cdc.gov/nchs/fastats/leading-causes-of-death.htm).

The dataset contains 303 rows and 14 columns with 13 features and 1 target variable. The target variable is the diagnosis of heart disease (angiographic disease status), and the value 0 is for no diagnosis of heart disease and the value 1 is for the diagnosis of heart disease.
The 13 features are as follows:
- Age
- Sex
- Chest pain type
- Resting blood pressure
- Serum cholestoral
- Fasting blood sugar
- Resting electrocardiographic
- Maximum heart rate achieved
- Exercise induced angina
- Oldpeak = ST depression induced by exercise relative to rest
- The slope of the peak exercise ST segment
- Number of major vessels 
- Thalassemia blood disorder

### Methodology

To answer the predictive question posed above, we will explore using different classifiers to predict the presence of heart disease, such as Logistic Regression, K-Nearest Neighbors, Decision Tree, and Support Vector Machine. We will also use the accuracy, precision, recall, and F1 score to judge how well the models work.

Before jumping into finding the best model and hyperparameters, we will be exploring the data to see if there are any missing values, outliers, and if there are any correlations between the features. We will also use bar charts, histograms, and scatter plots to look at the data and learn more about it.


### Sharing the results
All the reports, conclusions, and visualizations will be available from this github repository, in the form of Jupyter Notebooks, and hosted html files.


### License
Artifacts in this repository are [licensed](LICENSE) under the Attribution-NonCommercial-NoDerivatives 4.0 International, also known as CC BY-NC-ND 4.0.

### Contributing
We welcome contributions to this project. Please see our [contributing guidelines](CONTRIBUTING.md) for more information.

### Code of Conduct
Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.