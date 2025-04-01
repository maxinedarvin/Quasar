import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data_path = 'data_business_owner_imputed.csv'
print("Loading dataset...")
df = pd.read_csv(data_path)
print("Dataset loaded.")

df = df.drop(columns=['id']) # Drop columns irrelevant for the model

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Identify and remove outliers using the IQR method
def remove_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
    return df

df = remove_outliers(df, ['total_cash_inflow', 'total_cash_outflow'])

# Create new features
df['cash_inflow_outflow_ratio'] = df['total_cash_inflow'] / (df['total_cash_outflow'] + 1e-9) # Avoid division by zero

# Calculate and display the correlation matrix
print("Calculating correlation matrix...")
corr_matrix = df.corr()
print("Correlation matrix calculated.")
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Split the data into features (X) and target (y)
X = df.drop(columns=['total_cash_inflow', 'total_cash_outflow'])
y_inflow = df['total_cash_inflow']
y_outflow = df['total_cash_outflow']

# Split the data into training and testing sets
X_train, X_test, y_inflow_train, y_inflow_test = train_test_split(X, y_inflow, test_size=0.2, random_state=42)
_, _, y_outflow_train, y_outflow_test = train_test_split(X, y_outflow, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define and train the models with hyperparameter tuning
models = {
    'Random Forest Regressor': RandomForestRegressor(random_state=42, oob_score=True),
    'XGBoost Regressor': XGBRegressor(random_state=42)
}

param_grids = {
    'Random Forest Regressor': {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', None],
        'max_depth': [10, 20, 30]
    },
    'XGBoost Regressor': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.2]
    }
}

print("Evaluating models for total_cash_inflow:")
for name, model in models.items():
    if name in param_grids:
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_root_mean_squared_error')
        grid_search.fit(X_train, y_inflow_train)
        best_model = grid_search.best_estimator_
        print(f'Best parameters for {name}: {grid_search.best_params_}')
    else:
        best_model = model
        best_model.fit(X_train, y_inflow_train)

    y_inflow_pred = best_model.predict(X_test)
    mse_inflow = mean_squared_error(y_inflow_test, y_inflow_pred)
    rmse_inflow = np.sqrt(mse_inflow)
    mae_inflow = mean_absolute_error(y_inflow_test, y_inflow_pred)
    r2_inflow = r2_score(y_inflow_test, y_inflow_pred)
    print(f'{name} - MSE: {mse_inflow:.4f}')
    print(f'{name} - RMSE:{rmse_inflow:.4f}')
    print(f'{name} - MAE: {mae_inflow:.4f}')
    print(f'{name} - R^2 Score: {r2_inflow:.4f}')

    if name == 'Random Forest Regressor':
        print(f'{name} - OOB Score: {best_model.oob_score_:.4f}')

print("\nEvaluating models for total_cash_outflow:")
# Evaluate models for total_cash_outflow
for name, model in models.items():
    if name in param_grids:
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_root_mean_squared_error')
        grid_search.fit(X_train, y_outflow_train)
        best_model = grid_search.best_estimator_
        print(f'Best parameters for {name}: {grid_search.best_params_}')
    else:
        best_model = model
        best_model.fit(X_train, y_outflow_train)

    if name != 'Linear Regression':
        # Cross-validation
        cv_scores = cross_val_score(best_model, X_train, y_outflow_train, cv=5, scoring='r2')
        print(f'{name} - Cross-validated R^2 scores: {cv_scores}')
        print(f'{name} - Average R^2 score: {np.mean(cv_scores):.4f}')

    y_outflow_pred = best_model.predict(X_test)
    mse_outflow = mean_squared_error(y_outflow_test, y_outflow_pred)
    rmse_outflow = np.sqrt(mse_outflow)
    mae_outflow = mean_absolute_error(y_outflow_test, y_outflow_pred)
    r2_outflow = r2_score(y_outflow_test, y_outflow_pred)
    print(f'{name} - MSE: {mse_outflow:.4f}')
    print(f'{name} - RMSE: {rmse_outflow:.4f}')
    print(f'{name} - MAE: {mae_outflow:.4f}')
    print(f'{name} - R^2 Score: {r2_outflow:.4f}')

    if name == 'Random Forest Regressor':
        print(f'{name} - OOB Score: {best_model.oob_score_:.4f}')