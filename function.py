import numpy as np 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import os 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def read_file(file_path_train, file_path_test):
    """
    Read train and test data from CSV files.

    Parameters:
    - file_path_train: File path to the training data CSV file.
    - file_path_test: File path to the test data CSV file.

    Returns:
    - train_data: DataFrame containing the training data.
    - test_data: DataFrame containing the test data.
    """
    train_data = pd.read_csv(os.path.abspath(file_path_train))
    test_data = pd.read_csv(os.path.abspath(file_path_test))
    return train_data, test_data

def identify_columns_to_drop(df):
    """
    Identify columns to drop based on NaN values, constant predictors, and perfectly correlated predictors.

    Parameters:
    - df: DataFrame containing the data.

    Returns:
    - columns_to_drop: Array of column names to be dropped.
    """
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

def clean_data(train_data, test_data):
    """
    Clean train and test data by dropping identified columns.

    Parameters:
    - train_data: DataFrame containing the training data.
    - test_data: DataFrame containing the test data.

    Returns:
    - train_data_clean: DataFrame containing the clean training data.
    - test_data_clean: DataFrame containing the clean test data.
    """
    columns_to_drop = identify_columns_to_drop(train_data)
    train_data_clean = train_data.drop(columns=columns_to_drop, axis=1)
    test_data_clean = test_data.drop(columns=columns_to_drop, axis=1)
    return train_data_clean, test_data_clean 

def apply_one_hot_encoding(train_data, test_data, column_names):
    """
    Apply one-hot encoding to specified columns in train and test data.

    Parameters:
    - train_data: DataFrame containing the training data.
    - test_data: DataFrame containing the test data.
    - column_names: List of column names to apply one-hot encoding.

    Returns:
    - train_encoded: DataFrame with one-hot encoding applied to training data.
    - test_encoded: DataFrame with one-hot encoding applied to test data.
    """

    # Initialize the OneHotEncoder
    enc = OneHotEncoder(drop='first', sparse_output=False)

    # Create empty DataFrame for the new encoded columns
    all_encoded_train = pd.DataFrame()
    all_encoded_test = pd.DataFrame()

    # Iterate over each column and apply one-hot encoding on train and test
    for column_name in column_names:
        # Fit and transform the column
        train_encoded_data = enc.fit_transform(train_data[[column_name]])
        test_encoded_data = enc.fit_transform(test_data[[column_name]])

        # Create a DataFrame from the encoded data
        train_encoded = pd.DataFrame(train_encoded_data, columns=enc.get_feature_names_out([column_name]))
        test_encoded = pd.DataFrame(test_encoded_data, columns=enc.get_feature_names_out([column_name]))

        # Concatenate with the previously encoded columns
        all_encoded_train = pd.concat([all_encoded_train, train_encoded], axis=1)
        all_encoded_test = pd.concat([all_encoded_test, test_encoded], axis=1)

    # Concatenate the original DataFrame (minus the encoded columns) with the new DataFrame
    train_encoded = pd.concat([train_data.drop(column_names, axis=1), all_encoded_train], axis=1)
    test_encoded = pd.concat([test_data.drop(column_names, axis=1), all_encoded_test], axis=1)

    return train_encoded, test_encoded


# Not used here as we chose to use the training data with binary values
def standardize(df) :
    """
    Standardize the data by applying mean 0 and standard deviation 1 to every column.

    Parameters:
    - df: DataFrame containing the data.

    Returns:
    - df: Standardized DataFrame.
    """
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns = [df.columns])
    return df  

def get_processed_data(file_path):
    """
    Read, clean, and process data.

    Parameters:
    - file_path: File path to the data CSV file.

    Returns:
    - data: Processed DataFrame.
    """
    data = read_file(file_path)
    data = clean_data(data)
    #data = standardize(data)
    return data


def make_submission(df, file_path='submission.csv'):
    """
    Create a submission file from a DataFrame.

    Parameters:
    - df: DataFrame containing submission data.
    - file_path: File path to save the submission CSV file.
    """
    id = pd.DataFrame(np.arange(1,len(df)+1, dtype=int))
    submission_df = pd.concat([id,df], axis=1)
    submission_df.columns = ['ID','RT']
    submission_df.to_csv(os.path.abspath(file_path), index=False)
    print(f'Submission file saved to {file_path}')  

def tune_model(model, data, param_grid, submit=False):
    """
    Tune a machine learning model using grid search with cross-validation.

    Parameters:
    - model: The machine learning model to be tuned.
    - data: DataFrame containing the training data.
    - param_grid: dict, the parameter grid to search during tuning.
    - submit: bool, indicating whether the tuned model will be used for submission (default is False).

    Returns:
    - grid_search: GridSearchCV object with the tuned model.

    If submit is False:
    - Fits the grid search to a subset of the training data.

    If submit is True:
    - Fits the grid search to the entire training data.
    """
    
    # Set training data 
    if not submit:
        train_subset = int(len(data) * 0.8)
        data = data.iloc[:train_subset,:]
    print ("1")
    # Create a KFold cross-validator with 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    print ("2")

    # Create a grid search with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='neg_root_mean_squared_error')
    print ("3")

    # Fit the grid search to the data
    grid_search.fit(data.drop('RT', axis=1), data['RT'] )
    print ("4")

    return grid_search  


def test_error(y_test, y_pred, plot=False):
    """
    Calculate test error and optionally plot predictions against actual values.

    Parameters:
    - y_test: DataFrame containing the actual values.
    - y_pred: DataFrame containing the predicted values for the test set.
    - plot: Whether to plot predictions against actual values.

    Returns:
    - rmse: Root mean squared error for the test set.
    """
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
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'RMSE of {rmse}')
    return rmse


def apply_PCA(train_data, test_data, ratio) :
    """
    Apply Principal Component Analysis (PCA) to train and test data.

    Parameters:
    - train_data: DataFrame containing the training data.
    - test_data: DataFrame containing the test data.
    - ratio: The number of principal components to retain (float between 0 and 1 or integer).

    Returns:
    - train_PCA: DataFrame with PCA-transformed training data.
    - test_PCA: DataFrame with PCA-transformed test data.
    """

    # PCA for train data
    pca_train = PCA(n_components=ratio)
    train_PCA = pd.DataFrame(pca_train.fit_transform(train_data))

    # PCA for test data
    pca_test = PCA(n_components=ratio)
    test_PCA = pd.DataFrame(pca_test.fit_transform(test_data))
    return train_PCA, test_PCA

def summarize(cross_val_score:np.ndarray):
    """
    Summarize cross-validation scores, including average RMSE and RMSE for each fold.

    Parameters:
    - cross_val_score: Array of cross-validation scores (e.g., RMSE for each fold).

    Returns:
    - summary: Dictionary containing average RMSE and RMSE for each fold.
    """

    return {
        "avg_rmse" : cross_val_score.mean(),
        "rmse_per_fold": cross_val_score
    }


def split_data(data, target_column='RT', split_ratio=0.8):
    """
    Split the data into features (X) and target variable (y) using the specified split ratio.

    Parameters:
    - data: DataFrame, the input data.
    - target_column: str, the name of the target variable column.
    - split_ratio: float, the ratio to split the data into training and testing sets.

    Returns:
    - X_train, X_test, y_train, y_test: DataFrames, the separated training and testing sets.
    """
    train_subset = int(len(data) * split_ratio)

    X = data.drop([target_column], axis=1)
    y = data[target_column]

    X_train = X.iloc[:train_subset, :]
    X_test = X.iloc[train_subset:, :]
    y_train = y.iloc[:train_subset]
    y_test = y.iloc[train_subset:]

    return X_train, X_test, y_train, y_test

def model_test(train_data, test_data, model, submit=False, file_path='submission.csv', plot=False, target_column='RT'):
    """
    Train a machine learning model, evaluate it on a test set, and optionally make a submission.

    Parameters:
    - train_data: DataFrame, the training data.
    - test_data: DataFrame, the test data.
    - model: The machine learning model to be trained and evaluated.
    - submit: bool, indicating whether to make a submission (default is False).
    - file_path: str, the file path for submission (default is 'submission.csv').
    - plot: bool, indicating whether to plot evaluation results (default is False).
    - target_column: str, the name of the target variable column (default is 'RT').

    Returns:
    - y_pred: DataFrame, the predicted values.

    If submit is False:
    - Evaluates the model on a test subset and optionally plots the results.

    If submit is True:
    - Trains the model on all training data and makes a submission using the provided test data.
    """
    # Split data 
    X,y = train_data.drop([target_column], axis=1), train_data[target_column]
    X_train, X_test, y_train, y_test = split_data(train_data, target_column)

    if submit:
        model.fit(X,y) # train model on all training data
        y_pred = pd.DataFrame(model.predict(test_data)) # predict using test data
        make_submission(y_pred, file_path)
    else:
        model.fit(X_train,y_train) # train model on training subset
        y_pred = pd.DataFrame(model.predict(X_test)) # predict using test subset
        test_error(y_test,y_pred,plot) # test the error on the test subset
        
    return y_pred