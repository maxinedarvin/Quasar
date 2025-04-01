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
data_path = 'data_non_business_owner_imputed.csv'
print("Loading dataset...")
df = pd.read_csv(data_path)
print("Dataset loaded.")

# Drop columns with too many missing values or irrelevant for the model
df = df.drop(columns=['id'])

# Encode categorical variables
label_encoders = {}
encoded_df = df.copy()
for column in encoded_df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    encoded_df[column] = le.fit_transform(encoded_df[column])
    label_encoders[column] = le

def remove_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
    return df

encoded_df = remove_outliers(encoded_df, ['total_cash_inflow', 'total_cash_outflow'])
encoded_df['cash_inflow_outflow_ratio'] = encoded_df['total_cash_inflow'] / (encoded_df['total_cash_outflow'] + 1e-9) # Avoid division by zero

scoring_rules = {
    "citizenship": {"Filipino": 10, "Other": 5},
    "employment_and_or_business_information": {"Regular Employee": 5, "OFW/Seafarers": 4, "Project based": 3, "Probationary": 2},
    "years_in_residence": {"More than 2 years": 10, "1-2 years": 8, "6-12 months": 5, "Less than 6 months": 3},
    "additional_household_member_source_of_income": {"Employed": 5, "With Business": 4, "No other source of income": 3},
    "do_you_have_a_co_borrower": {"Yes": 5, "No, please prepare postdated checks for your monthly amortization": 3},
    "my_spouse_is_my_co_borrower": {"Yes": 3, "No": 0},
    "years_in_residence": {"More than 2 years": 5, "1-2 years": 4, "6-12 months": 3, "Less than 6 months": 0},
    "home_ownership": {"Owned": 10, "Free": 8, "Rented": 5, "Others": 3},
    "agency_years": {"More than 5 years": 5, "1-5 years": 4, "Less than a year": 3},
    "marital_status": {"Married": 5, "Widowed": 4, "Single": 3, "Separated": 3},
    "no_of_children": {"0-1": 5, "2-3": 4, "more than 3": 3},
    "with_children_attending_school": {"Yes": 5, "No": 3},
    "position": {"Professional/Managerial/Officer": 5, "Professional/Non-Officer/Skilled": 3, "Others": 1},
    "do_you_have_deposit_or_credit_cards_with_other_bank": {"Yes": 5, "No": 3}
}

def calculate_credit_score(row, scoring_rules):
    score = 0
    for column, rules in scoring_rules.items():
        value = row[column]
        score += rules.get(value, 0)
    return score

# Apply the function to calculate credit scores
df['credit_score'] = df.apply(calculate_credit_score, scoring_rules=scoring_rules, axis=1)

# Save the updated DataFrame to a CSV file
output_path = 'data_non_business_owner_imputed_with_credit_score.csv'
df.to_csv(output_path, index=False)
print(f"Updated dataset saved to {output_path}.")

print(df.head())

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le
    
print("Calculating correlation matrix...")
corr_matrix = df.corr()
print("Correlation matrix calculated.")
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Split the data into features (X) and target (y)
X = df.drop(['credit_score'], axis=1)
y = df['credit_score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

print("Evaluating models for credit_score:")
for name, model in models.items():
    if name in param_grids:
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_root_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f'Best parameters for {name}: {grid_search.best_params_}')
    else:
        best_model = model
        best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    mse_inflow = mean_squared_error(y_test, y_pred)
    rmse_inflow = np.sqrt(mse_inflow)
    mae_inflow = mean_absolute_error(y_test, y_pred)
    r2_inflow = r2_score(y_test, y_pred)
    print(f'{name} - MSE: {mse_inflow:.4f}')
    print(f'{name} - RMSE:{rmse_inflow:.4f}')
    print(f'{name} - MAE: {mae_inflow:.4f}')
    print(f'{name} - R^2 Score: {r2_inflow:.4f}')

    if name == 'Random Forest Regressor':
        print(f'{name} - OOB Score: {best_model.oob_score_:.4f}')