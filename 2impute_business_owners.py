import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder
import numpy as np

data_path = 'data_business_owner.csv'
print("Loading dataset...")
df = pd.read_csv(data_path)
print("Dataset loaded.")
print(df.head())

# df['id'] = df['id'].astype(int)
# df['gender'] = df['gender'].astype('object')
# df['citizenship'] = df['citizenship'].astype('object')
# df['region'] = df['region'].astype('object')
# df['province'] = df['province'].astype('object')
# df['city_municipality'] = df['city_municipality'].astype('object')
# df['employment_and_or_business_information'] = df['employment_and_or_business_information'].astype('object')
# df['additional_household_member_source_of_income'] = df['additional_household_member_source_of_income'].astype('object')
# df['do_you_have_a_co_borrower'] = df['do_you_have_a_co_borrower'].astype('object')
# df['my_spouse_is_my_co_borrower'] = df['my_spouse_is_my_co_borrower'].astype('object')
# df['business_gross_sales'] = df['business_gross_sales'].astype(float)
# df['total_cash_inflow'] = df['total_cash_inflow'].astype(float)
# df['total_cash_outflow'] = df['total_cash_outflow'].astype(float)
# # df['net_income'] = df['net_income'].astype(float)
# df['years_in_residence'] = df['years_in_residence'].astype('object')
# df['home_ownership'] = df['home_ownership'].astype('object')
# df['business_tenure'] = df['business_tenure'].astype('object')
# df['ownership_of_business_premises'] = df['ownership_of_business_premises'].astype('object')
# df['number_of_paid_employees'] = df['number_of_paid_employees'].astype('object')
# df['business_nature'] = df['business_nature'].astype('object')
# df['marital_status'] = df['marital_status'].astype('object')
# df['no_of_children'] = df['no_of_children'].astype('object')
# df['with_children_attending_school'] = df['with_children_attending_school'].astype('object')
# df['outstanding_obligation_other_loans'] = df['outstanding_obligation_other_loans'].astype(float)
# df['do_you_have_deposit_or_credit_cards_with_other_bank'] = df['do_you_have_deposit_or_credit_cards_with_other_bank'].astype('object')

mode_impute_cols = ['gender', 'my_spouse_is_my_co_borrower', 'with_children_attending_school']  # just 1 missing data
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

imputed_data_path = 'data_business_owner_imputed.csv'
df.to_csv(imputed_data_path, index=False)
print(f"Imputed dataset saved to {imputed_data_path}")