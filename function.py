import numpy as np 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import os 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, ChemicalFeatures
from rdkit import RDConfig

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

def analyze_smiles_column(dataframe1, smiles_column):
    """
    Analyze the SMILES column of a DataFrame and extract molecular descriptors.

    Parameters:
    - dataframe1 (pd.DataFrame): DataFrame containing the SMILES column.
    - smiles_column (str): Name of the column containing SMILES strings.

    Returns:
    pd.DataFrame: DataFrame with extracted molecular descriptors.
    """

    dataframe = dataframe1.copy()
    dataframe['Molecule'] = dataframe[smiles_column].apply(Chem.MolFromSmiles)

    # Basic information about the molecules
    dataframe['Num_Atoms'] = dataframe['Molecule'].apply(Descriptors.HeavyAtomCount)
    dataframe['Num_Bonds'] = dataframe['Molecule'].apply(Descriptors.NumRotatableBonds)
    dataframe['Num_Rings'] = dataframe['Molecule'].apply(Descriptors.RingCount)
    dataframe['MolWeight'] = dataframe['Molecule'].apply(Descriptors.MolWt)
    dataframe['LogP'] = dataframe['Molecule'].apply(Descriptors.MolLogP)
    dataframe['TPSA'] = dataframe['Molecule'].apply(Descriptors.TPSA)

    # Additional molecular descriptors
    dataframe['Num_HydrogenAcceptors'] = dataframe['Molecule'].apply(Descriptors.NumHAcceptors)
    dataframe['Num_HydrogenDonors'] = dataframe['Molecule'].apply(Descriptors.NumHDonors)
    dataframe['Num_AromaticRings'] = dataframe['Molecule'].apply(Descriptors.NumAromaticRings)
    dataframe['Num_SaturatedRings'] = dataframe['Molecule'].apply(Descriptors.NumSaturatedRings)
    dataframe['Num_AliphaticRings'] = dataframe['Molecule'].apply(Descriptors.NumAliphaticRings)
    dataframe['Num_AliphaticCarbocycles'] = dataframe['Molecule'].apply(Descriptors.NumAliphaticCarbocycles)

    # Molecular fingerprints
    dataframe['FunctionalGroups'] = dataframe['Molecule'].apply(lambda x: len(ChemicalFeatures.BuildFeatureFactory(
        os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')).GetFeaturesForMol(x)))
    dataframe['Fingerprint'] = dataframe['Molecule'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2))

    return dataframe[['Num_Atoms', 'Num_Bonds', 'Num_Rings', 'MolWeight', 'TPSA',
                       'Num_HydrogenAcceptors', 'Num_HydrogenDonors', 'Num_AromaticRings',
                       'Num_SaturatedRings', 'Num_AliphaticRings', 'Num_AliphaticCarbocycles',
                       'FunctionalGroups']]

def add_cddd(df, smiles_column):
    """
    Add Chemical Diversity Descriptors (CDDD) to a DataFrame based on the 'smiles' column.

    Parameters:
    - df (pd.DataFrame): DataFrame to which CDDD will be added.
    - smiles_column (str): Name of the column containing SMILES strings.

    Returns:
    pd.DataFrame: DataFrame with added CDDD.
    """
    cddd_df =  pd.read_csv(os.path.abspath('cddd.csv'))

    # Merge based on the 'smiles' column
    merged_df = pd.merge(df, cddd_df, on=smiles_column, how='left')

    # Drop columns in common with df
    merged_df = merged_df.drop(df.columns, axis=1)
    
    return merged_df

def process_data(train_data, test_data, include_cdd=False):
    """
    Process and clean training and test data.

    Parameters:
    - train_data (pd.DataFrame): DataFrame containing training data.
    - test_data (pd.DataFrame): DataFrame containing test data.
    - include_cdd (bool): Flag to include or exclude CDD. Default is False.
    """
      
    smiles_train, smiles_test  = analyze_smiles_column(train_data, 'SMILES'), analyze_smiles_column(test_data, 'SMILES')
    smiles_train, smiles_test = clean_data(smiles_train, smiles_test)
    smiles_train, smiles_test  = standardize(smiles_train), standardize(smiles_test)
    
    if include_cdd:
        cddd_train, cddd_test = add_cddd(train_data, 'SMILES'), add_cddd(test_data, 'SMILES')
        cddd_train, cddd_test = clean_data(cddd_train, cddd_test)
        cddd_train, cddd_test = standardize(cddd_train), standardize(cddd_test)

    # Drop categorical values
    test_data = test_data.drop(['SMILES', 'mol','Compound'], axis=1)
    train_data = train_data.drop(['SMILES', 'mol','Compound'], axis=1)

    # Columns to encode
    columns_to_encode = ['Lab']

    # OneHotEncoder on train_data and test_data
    train_data, test_data = apply_one_hot_encoding(train_data, test_data, columns_to_encode)

    # Concate train, smiles and cdd
    if include_cdd:
        train_data, test_data  = pd.concat([train_data, smiles_train, cddd_train], axis=1), pd.concat([test_data, smiles_test, cddd_test], axis=1)
    else :
        train_data, test_data = pd.concat([train_data, smiles_train], axis=1), pd.concat([test_data, smiles_test], axis=1)

    # Fill NA values by 0
    test_data = test_data.fillna(0)  
    train_data = train_data.fillna(0) 

    # Clean the resulting DataFrame
    train_data, test_data = clean_data(train_data, test_data)

    train_data.columns = [str(col) for col in train_data.columns]
    test_data.columns = [str(col) for col in test_data.columns] 

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
        df.dropna(inplace=True)

    # Identifies constant predictors
    thr = 0.1
    constant_columns = df.columns[df.std(axis=0, numeric_only=True) <= thr]
    constant_columns = list(constant_columns)
    
    # Identifies perfectly correlated predictors
    correlation_matrix = df.corr().abs()
    upper_triangle_mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    upper_triangle = correlation_matrix.where(upper_triangle_mask)
    correlated_columns = [column for column in upper_triangle.columns if any(upper_triangle[column] ==1)]

    # Combine constant and correlated columns
    columns_to_drop = constant_columns + correlated_columns

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

    # Drop columns
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

# Used to find hyperparameters 
def tune_model(model, X_train, y_train , param_grid, submit=False):
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
    
        
    # Create a KFold cross-validator with 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Create a grid search with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='neg_root_mean_squared_error')

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    print("Best hyperparameters:", grid_search.best_params_)
   
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

    # RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'test error of {rmse}')

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

    # PCA on train data
    pca_train = PCA(n_components=ratio)
    train_PCA = pd.DataFrame(pca_train.fit_transform(train_data))

    # PCA on test data
    pca_test = PCA(n_components=ratio)
    test_PCA = pd.DataFrame(pca_test.fit_transform(test_data))

    return train_PCA, test_PCA

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, plot=False):
    """
    Train and evaluate a machine learning model using cross-validation.

    Parameters:
    - model: The machine learning model to be trained and evaluated.
    - X_train: The features of the training data.
    - y_train: The target variable of the training data.
    - X_test: The features of the test data.
    - y_test : The target variable of the test data.
    - plot: bool, indicating whether to plot test error (default is False).

    """
    # Create a KFold cross-validator with 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Perform cross-validation and calculate the mean squared error
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')

    # Print cross-validation RMSE scores
    print("Cross-Validation RMSE Scores:", -cv_scores)
    print("Mean CV RMSE:", -cv_scores.mean())

    # Train the model on the training set
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = pd.DataFrame(model.predict(X_test))

    # Test the model
    test_error(y_test, y_pred, plot)
