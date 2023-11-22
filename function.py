import numpy as np 
import pandas as pd
import os 
from sklearn.preprocessing import StandardScaler


def read_file(file_path):
    df = pd.read_csv(os.path.abspath(file_path), header=0)
    # Vérifier si 'RT' est dans les colonnes et le sélectionner si présent
    columns_to_keep = [col for col in df.columns if col.startswith('ECFP')]
    if 'RT' in df.columns:
        columns_to_keep = ['RT'] + columns_to_keep
    df = df[columns_to_keep]
    return df


def clean_data(df):
    #drop nan values
    if (df.isna().any().any()):
        df.dropna(inplace = True)
    #removes constant predictors  
    df = df.loc[:, np.std(df, axis=0) != 0]
    #removes perfectly correlated predictors 
    correlation = np.array(df.corr().values) #compute correlation
    correlation = np.triu(correlation, k=0) # remove values in double
    np.fill_diagonal(correlation,0) # remove "self column" correlation
    df = df.drop(df.columns[np.where(correlation==1)[1]], axis=1) #remove one of the column with exact correlation
    return df

#The transformed data has now mean 0 and standard deviation 1 in every column.
def standardize(df) :
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns = [df.columns])
    return df

def get_processed_data(file_path):
    data = read_file(file_path)
    data = standardize(data)
    data = clean_data(data)
    return data

    

