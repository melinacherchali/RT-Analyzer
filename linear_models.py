from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import function as f 

# Load data
train_data = f.read_file('train.csv')
test_data = f.read_file('test.csv')

# drop categorical values
test_data = test_data.drop(['Compound', 'SMILES', 'mol'], axis=1)
train_data = train_data.drop(['Compound', 'SMILES','mol'], axis=1)

#OneHotEncoder on X
train_data_encoded = f.apply_one_hot_encoding(train_data, 'Lab')

#OneHotEncoder on test_data
test_data_encoded = f.apply_one_hot_encoding(test_data, 'Lab')

# identify columns to drop from the training data
columns_to_drop = f.identify_columns_to_drop(train_data_encoded)

# Then, clean both train and test data using these columns
train_data_cleaned = f.clean_data(train_data_encoded, columns_to_drop)
test_data_cleaned = f.clean_data(test_data_encoded, columns_to_drop)

# separate the features (X) and the target variable (y)
train_subset = 2800
X = train_data_cleaned .drop(['RT'], axis=1)
X_subset = X.iloc[:train_subset, :]
y = train_data_cleaned['RT'].iloc[:train_subset]

""" --------------------------------------------------- Linear Regression --------------------------------------------------- """

"""
#does not return correct predictions

# Define model
model = LinearRegression()

# fit the model to the entire training set if you plan to make predictions
model.fit(X_subset, y)

# make predictions
predictions = model.predict(X.iloc[train_subset:, :])
print (predictions)  """

""" # Plot the predictions against actual values
plt.figure()
plt.scatter(predictions,  train_data['RT'].iloc[train_subset:], label="Predictions vs Actual Values")
plt.plot(np.arange(len(predictions)), np.arange(len(predictions)), c="black", ls="dashed", label="y=x")
plt.xlabel("Predictions")
plt.ylabel("Actual Values")
plt.legend()
plt.show()  """


"""--------------------------------------------------- Ridge Regression ---------------------------------------------------""" 
"""
# Define the RidgeRegressor model
model = Ridge()

# Define the parameter grid for hyperparameter tuning
param_grid = {'alpha': np.logspace(-5, 7, 100)} # this range finds better alpha

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_subset, y)

# Get the best model from the grid search
best_model = grid_search.best_estimator_
#print(grid_search.best_estimator_.alpha)

# Fit the best model to a subset of the data
train_subset = 2800  # Define the size of the training subset
best_model.fit(X.iloc[:train_subset, :], y.iloc[:train_subset])

# Make predictions on the remaining data
predictions = best_model.predict(X.iloc[train_subset:, :])
print(predictions)

# Plot the predictions against actual values
plt.figure()
plt.scatter(predictions, y.iloc[train_subset:], label="Predictions vs Actual Values")
plt.plot(np.arange(len(predictions)), np.arange(len(predictions)), c="black", ls="dashed", label="y=x")
plt.xlabel("Predictions")
plt.ylabel("Actual Values")
plt.xlim(0, 6)
plt.ylim(0, 4)
plt.legend()
plt.show()  """