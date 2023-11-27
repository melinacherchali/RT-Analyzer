
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

import numpy as np
import function as f 
import pandas as pd


def summarize(cross_val_score:np.ndarray):
    return {
        "avg_rmse" : cross_val_score.mean(),
        "rmse_per_fold": cross_val_score
    }


# Load data
train_data = f.read_file('train.csv')
test_data = f.read_file('test.csv')

# Drop categorical values
test_data = test_data.drop(['SMILES', 'mol','Compound'], axis=1)
train_data = train_data.drop(['SMILES','mol','Compound'], axis=1)

# Columns to encode
columns_to_encode = ['Lab']

# OneHotEncoder on train_data
train_data_encoded = f.apply_one_hot_encoding(train_data, columns_to_encode)

# OneHotEncoder on test_data
test_data_encoded = f.apply_one_hot_encoding(test_data, columns_to_encode)

# Identify columns to drop from the training data
columns_to_drop = f.identify_columns_to_drop(train_data_encoded)

# Clean both train and test data using these columns
train_data = f.clean_data(train_data_encoded, columns_to_drop)
test_data = f.clean_data(test_data_encoded, columns_to_drop)

# Separate the features X and the target variable y
train_subset = int(len(train_data) * 0.8)

X = train_data.drop(['RT'], axis=1)
y = train_data['RT']

X_train = X.iloc[:train_subset, :]
X_test = X.iloc[train_subset:, :]
y_train = y.iloc[:train_subset]
y_test = y.iloc[train_subset:]


# Define a custom scorer for RMSE
def rmse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean() ** 0.5

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_model = RandomForestRegressor(n_estimators=500)

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Train the model on the training set
rf_model.fit(X, y)
# rf_model.fit(X_train,y_train)


# Evaluate the model on the test set
y_pred = pd.DataFrame(rf_model.predict(test_data))

# y_pred = pd.DataFrame(rf_model.predict(X_test))
# f.test_error(y_test, y_pred, True)
f.make_submission(y_pred, "random_forest_sub.csv")