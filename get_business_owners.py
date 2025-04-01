import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

data_df = pd.read_csv('data_v2_LESS_IMAGES.csv')
print("Data loaded successfully.")

data_df.dropna(subset=[data_df.columns[19], data_df.columns[20], data_df.columns[21]], inplace=True)

data_df.loc[data_df['employment_and_or_business_information'].isna(), 'employment_and_or_business_information'] = data_df['employment_status']

# data_df['employment_and_or_business_information'].replace(['Locally Employed', 'Locally Employed with Business'], 'Regular Employee', inplace=True)

data_df['additional_household_member_source_of_income'].replace(['Employed', 'With Business'], 'Yes', inplace=True)
data_df['additional_household_member_source_of_income'].replace(['No other source of income'], 'No', inplace=True)

data_df['do_you_have_a_co_borrower'].replace(['No, please prepare postdated checks for your monthly amortization'], 'No', inplace=True)

data_df['gender'].replace(['F', 'female'], 'Female', inplace=True)

business_owner_df = data_df[data_df['employment_and_or_business_information'] == 'Business Owner']
non_business_owner_df = data_df[data_df['employment_and_or_business_information'] != 'Business Owner']

columns_to_drop_business_owner = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 67, 69, 73, 77, 78, 79, 80, 81, 83, 84, 86]
business_owner_df.drop(business_owner_df.columns[columns_to_drop_business_owner], axis=1, inplace=True)

columns_to_drop_non_business_owner = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 67, 68, 69, 70, 71, 72, 79, 80, 81, 83, 84, 86]
non_business_owner_df.drop(non_business_owner_df.columns[columns_to_drop_non_business_owner], axis=1, inplace=True)

business_owner_df.to_csv('/Downloads/Quasar/data_business_owner.csv', index=False)
non_business_owner_df.to_csv('/Downloads/Quasar/data_non_business_owner.csv', index=False)

print("Data processing and export completed successfully.")