import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder
import numpy as np

data_path = 'data_non_business_owner.csv'
print("Loading dataset...")
df = pd.read_csv(data_path)
print("Dataset loaded.")
print(df.head())

print(df.isna().sum())

df = df.dropna(subset=['years_in_residence']) # drop rows with missing values in these columns, as they have missing values in other columns too
print(df.isna().sum())

mode_impute_cols = ['gender', 'do_you_have_a_co_borrower', 'my_spouse_is_my_co_borrower', 'agency_years', 'with_children_attending_school', 'how_many_years_in_the_company', 'position']  # less than 10 missing data
print("Performing mode imputation for columns:", mode_impute_cols)
mode_imputer = SimpleImputer(strategy='most_frequent')
df[mode_impute_cols] = mode_imputer.fit_transform(df[mode_impute_cols])

print("Data after mode imputation:")
print(df[mode_impute_cols].head())

print("Encoding categorical columns to numerical values...")
categorical_cols = df.select_dtypes(['object']).columns
label_encoders = {}
for col in categorical_cols:
    print(f"Encoding column: {col}")
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"Unique values in {col} after encoding: {df[col].unique()}")

print("Checking for 'nan' strings in 'do_you_have_deposit_or_credit_cards_with_other_bank' and converting to NaN...")
df['do_you_have_deposit_or_credit_cards_with_other_bank'].replace(2, np.nan, inplace=True)

print("Missing values before imputation:")
print(df.isnull().sum())

print("Performing KNN imputation...")
imputer = KNNImputer(n_neighbors=25)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print("Data after KNN imputation:")
print(df.head())

for col in categorical_cols:
    le = label_encoders[col]
    df[col] = le.inverse_transform(df[col].round().astype(int))
    print(f"Unique values in {col} after decoding: {df[col].unique()}")

print("Unique values of 'do_you_have_deposit_or_credit_cards_with_other_bank' after imputation:")
print(df['do_you_have_deposit_or_credit_cards_with_other_bank'].unique())

imputed_data_path = 'data_non_business_owner_imputed.csv'
df.to_csv(imputed_data_path, index=False)
print(f"Imputed dataset saved to {imputed_data_path}")