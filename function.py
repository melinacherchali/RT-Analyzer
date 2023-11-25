import numpy as np 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


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

def apply_one_hot_encoding(df, column_names):
    # Initialize the OneHotEncoder
    enc = OneHotEncoder(drop='first', sparse_output=False)

    # Create an empty DataFrame for the new encoded columns
    all_encoded_df = pd.DataFrame()

    # Iterate over each column and apply one-hot encoding
    for column_name in column_names:
        # Fit and transform the column
        encoded_data = enc.fit_transform(df[[column_name]])

        # Create a DataFrame from the encoded data
        encoded_df = pd.DataFrame(encoded_data, columns=enc.get_feature_names_out([column_name]))

        # Concatenate with the previously encoded columns
        all_encoded_df = pd.concat([all_encoded_df, encoded_df], axis=1)

    # Concatenate the original DataFrame (minus the encoded columns) with the new DataFrame
    df_encoded = pd.concat([df.drop(column_names, axis=1), all_encoded_df], axis=1)

    return df_encoded

""" def apply_one_hot_encoding(df, column_name):
    # Initialize the OneHotEncoder
    enc = OneHotEncoder(drop='first', sparse_output=False)

    # Fit and transform the specified column
    encoded_data = enc.fit_transform(df[[column_name]])

    # Create a DataFrame from the encoded data
    encoded_df = pd.DataFrame(encoded_data, columns=enc.get_feature_names_out([column_name]))

    # Concatenate the original DataFrame (minus the encoded column) with the new DataFrame
    df_encoded = pd.concat([df.drop([column_name], axis=1), encoded_df], axis=1)

    return df_encoded """


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

def tune_model(model, data):

    # Create a KFold cross-validator with 20 folds
    kf = KFold(n_splits=20, shuffle=True, random_state=42)

    # Define the parameter grid for tuning alpha
    param_grid = {
        'alpha': np.logspace(-1, 5, num =30)
    }

    # Create a grid search with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='neg_root_mean_squared_error')

    # Fit the grid search to the data
    grid_search.fit(data.drop('RT', axis=1), data['RT'] )

    return grid_search  


def test_error(y_test, y_pred, plot=False):
    # Plot the predictions against actual values
    if plot:
        plt.figure()
        plt.scatter(y_pred,  y_test, label="Predictions vs Actual Values")
        plt.plot(np.arange(len(y_pred)), np.arange(len(y_test)), c="black", ls="dashed", label="y=x")
        plt.xlabel("Predictions")
        plt.ylabel("Actual Values")
        plt.xlim(0, 30)
        plt.ylim(0, 30)
        plt.legend()
        plt.show() 
    mse = mean_squared_error(y_test, y_pred)
    print(f'Test error of {mse}')
    return mse
