import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

data_df = pd.read_csv('/Downloads/Quasar/data_v2_LESS_IMAGES.csv')
print("Data loaded successfully.")

# Drop rows with empty values in columns 19-21, if we're using location data. if not, drop columns 19-21
# data_df.dropna(subset=[data_df.columns[19], data_df.columns[20], data_df.columns[21]], inplace=True)

data_df['employment_and_or_business_information'].fillna(data_df['employment_status'], inplace=True)

data_df['employment_and_or_business_information'].replace(
    {'Locally Employed': 'Regular Employee', 'Locally Employed with Business': 'Regular Employee', 'Locally Employed with Business ': 'Regular Employee', 'OFW/Seafarers with Business': 'OFW/Seafarers'},
    inplace=True
)
data_df['gender'].replace({'F': 'Female', 'female': 'Female', 'FEMALE': 'Female'}, inplace=True)

data_df.loc[data_df['education'] != 0, 'education_expenses'] = data_df['education']
data_df.loc[data_df['rental'] != 0, 'rental_expenses'] = data_df['rental']
data_df.loc[data_df['utility_bills'] != 0, 'utilities'] = data_df['utility_bills']
data_df.loc[data_df['food_and_grocery'] != 0, 'food_expenses'] = data_df['food_and_grocery']

df_business_owner = data_df[data_df['employment_and_or_business_information'] == 'Business Owner']
df_non_business_owner = data_df[data_df['employment_and_or_business_information'] != 'Business Owner']

columns_to_drop_business_owner = data_df.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 47, 48, 49, 50, 67, 69, 77, 78, 79, 80, 81, 83, 84, 86]]
columns_to_drop_non_business_owner = data_df.columns[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 47, 48, 49, 50, 67, 68, 69, 70, 71, 72, 79, 80, 81, 83, 84, 86]]


df_business_owner.drop(columns_to_drop_business_owner, axis=1, inplace=True)
df_non_business_owner.drop(columns_to_drop_non_business_owner, axis=1, inplace=True)

df_business_owner.to_csv('/Downloads/Quasar/data_business_owner.csv', index=False)
df_non_business_owner.to_csv('/Downloads/Quasar/data_non_business_owner.csv', index=False)

print("DataFrames processed and exported successfully.")