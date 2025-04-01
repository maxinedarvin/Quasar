# README
This repository contains files for predicting total_cashflow_input, total_cashflow_output, and the credit_score of 2500+ banking customers based on loaning data and demographic data provided by a local Philippine bank that wishes to remain undisclosed. This is for a business pitch to create an AI model and website to automate customer inputs and profiling to significantly speed up loan processing and approval. The single-digit prefixes before each file name indicate the order in which the .py files are to be run, and the corresponding step in which the .csv was created. 

### NOTE: The data_v2_LESS_IMAGES.csv dataset from Step 1 cannot be uploaded publicly.

## 1: Data Cleaning
To organize the data, I merged and kept all non-empty data of redundant columns onto their duplicates before dropping the extra column, dropped columns with a significantly large amount of NaN values that made it unfit for imputing, dropped columns with zero variance, and dropped columns with no relation to predictive model. I simplified repetitive column values and fixed typographical errors. Lastly, I separated rows with 'Business Owner' values as these have column values unique to themselves as did non- 'Business Owner' values.

## 2: Data Imputation
For data imputation of demographic variables, I implemented mode imputation and KNNImputer depending on the quantity of NaN data relative to the quantity of non-NaN entries of a column. For KNNImputer, I used LabelEncoder to first encode all categorical variables into numerical values before implementing the model.

## 3: Data Prediction, Outlier Removal, Feature Engineering
For credit score, I defined a scoring function depending on the unique values of each column that gave higher points to individuals for traits that imply better financial capabilities. I ran this function to aggregate a new column 'credit_score' that will be set as the target variable of the predictive model. For cashflow, I set the total_cash_inflow and total_cash_outflow columns as target variables. I used IQR to detect and remove outlier data of target variables for both Python files for better model generalization. 

Additionally, I feature engineered a new variable: cash_inflow_outflow_ratio. I printed a correlation matrix to see the correlation coefficients of all variables and look for potential data leaks or errors.

Afterward, I used a simple implementation of RandomForestRegressor from sklearn.ensemble and XGBoostRegressor from xgboost. Both models underwent five-fold cross-validation, an 80:20 train-test split, and GridSearchCV for hyperparameter optimization. Both models were evaluated using MSE, RMSE, MAE, and R^2 scores. The Random Forest model additionally computed for OOB score. All scores of both models were 0.80 and above.


## Recommendation
Additional modifications may be done in the following areas: scoring values, error metrics, imputations, grid values, to name a few. I highly recommend a greater amount of data samples in future implementations.
