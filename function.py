import numpy as np 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os 
from sklearn.preprocessing import StandardScaler


def read_file(file_path):
    df = pd.read_csv(os.path.abspath(file_path))
    return df

def identify_columns_to_drop(df):
    # Drop nan values
    if df.isna().any().any():
        df = df.dropna(inplace = True)

    # Identifies constant predictors
    constant_columns = df.columns[df.std(axis=0, numeric_only=True) == 0]

    # Identifies perfectly correlated predictors
    correlation = np.array(df.corr().values)
    correlation = np.triu(correlation, k=1)
    correlated_columns = df.columns[np.where(correlation == 1)[0]]

    # Combine constant and correlated columns
    columns_to_drop = np.union1d(constant_columns, correlated_columns)
    return columns_to_drop

def clean_data(df, columns_to_drop):
    return df.drop(columns=columns_to_drop, axis=1)


def apply_one_hot_encoding(df, column_name):
    # Initialize the OneHotEncoder
    enc = OneHotEncoder(drop='first', sparse_output=False)

    # Fit and transform the specified column
    encoded_data = enc.fit_transform(df[[column_name]])

    # Create a DataFrame from the encoded data
    encoded_df = pd.DataFrame(encoded_data, columns=enc.get_feature_names_out([column_name]))

    # Concatenate the original DataFrame (minus the encoded column) with the new DataFrame
    df_encoded = pd.concat([df.drop([column_name], axis=1), encoded_df], axis=1)

    return df_encoded

""" def clean_data(df):
    # Drop nan values
    if (df.isna().any().any()):
        df.dropna(inplace = True)
    # Removes constant predictors  
    df = df.loc[:, np.std(df, axis=0) != 0]
    # Removes perfectly correlated predictors 
    correlation = np.array(df.corr().values) 
    correlation = np.triu(correlation, k=0)
    np.fill_diagonal(correlation,0) 
    df = df.drop(df.columns[np.where(correlation==1)[1]], axis=1)
    return df """

""" # The transformed data has now mean 0 and standard deviation 1 in every column.
def standardize(df) :
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns = [df.columns])
    return df """

def get_processed_data(file_path):
    data = read_file(file_path)
    data = clean_data(data)
    #data = standardize(data)
    return data


def make_submission(df, file_path='submission.csv'):
    id = pd.DataFrame(np.arange(1,len(df)+1, dtype=int))
    submission_df = pd.concat([id,df], axis=1)
    submission_df.columns = ['ID','RT']
    submission_df.to_csv(os.path.abspath(file_path), index=False)
    print(f'Submission file saved to {file_path}')    