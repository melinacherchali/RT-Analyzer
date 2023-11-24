from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import function as f 

""" --------------------------------------------------- Linear Regression --------------------------------------------------- """
# Load data
train_data = f.read_file('train.csv')
test_data = f.read_file('test.csv')
#train_data = f.get_processed_data('train.csv')
#test_data = f.get_processed_data('test.csv')

columns_to_drop = ['Compound', 'SMILES', 'RT', 'mol']
test_data = test_data.drop(['Compound', 'SMILES', 'mol'], axis=1)

X = train_data.drop(columns=columns_to_drop)

#OneHotEncoder on X
enc = OneHotEncoder(drop='first', sparse_output=False)
lab_encoded = enc.fit_transform(X[['Lab']])
lab_encoded_df = pd.DataFrame(lab_encoded, columns=enc.get_feature_names_out(['Lab']))
data_X_encoded = pd.concat([X.drop(['Lab'], axis=1), lab_encoded_df], axis=1)

#OneHotEncoder on test
test_lab_encoded = enc.transform(test_data[['Lab']])
test_lab_encoded_df = pd.DataFrame(test_lab_encoded, columns=enc.get_feature_names_out(['Lab']))
test_data_encoded = pd.concat([test_data.drop(['Lab'], axis=1), test_lab_encoded_df], axis=1)

# separate the features (X) and the target variable (y)
train_subset = 2800
X = data_X_encoded.iloc[:train_subset, :]
y = train_data['RT'].iloc[:train_subset]

# Define model
model = LinearRegression()

# fit the model to the entire training set if you plan to make predictions
model.fit(X, y)

# make predictions
predictions = model.predict(data_X_encoded.iloc[train_subset:, :])

# Plot the predictions against actual values
plt.figure()
plt.scatter(predictions,  train_data['RT'].iloc[train_subset:], label="Predictions vs Actual Values")
plt.plot(np.arange(len(predictions)), np.arange(len(predictions)), c="black", ls="dashed", label="y=x")
plt.xlabel("Predictions")
plt.ylabel("Actual Values")
plt.xlim(0, 30)
plt.ylim(0, 30)
plt.legend()
plt.show() 


"""--------------------------------------------------- Ridge Regression --------------------------------------------------- 

# Load data
train_data = f.get_processed_data('train.csv')
test_data = f.get_processed_data('test.csv')

# Split features and target variable
X = train_data.sort_index(axis=1).drop('RT', axis=1)  
y_pred= train_data['RT']

# Define the RidgeRegressor model
model = Ridge()

# Define the parameter grid for hyperparameter tuning
param_grid = {'alpha': np.logspace(-5, 7, 100)} # this range finds better alpha

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y_pred)

# Get the best model from the grid search
best_model = grid_search.best_estimator_
print(grid_search.best_estimator_.alpha)

# Fit the best model to a subset of the data
train_subset = 2800  # Define the size of the training subset
best_model.fit(X.iloc[:train_subset, :], y_pred.iloc[:train_subset])

# Make predictions on the remaining data
predictions = best_model.predict(X.iloc[train_subset:, :])

# Plot the predictions against actual values
plt.figure()
plt.scatter(predictions, y_pred.iloc[train_subset:], label="Predictions vs Actual Values")
plt.plot(np.arange(len(predictions)), np.arange(len(predictions)), c="black", ls="dashed", label="y=x")
plt.xlabel("Predictions")
plt.ylabel("Actual Values")
plt.xlim(0, 6)
plt.ylim(0, 4)
plt.legend()
plt.show() """