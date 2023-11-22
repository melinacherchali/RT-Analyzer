from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import function as f 

#load_data
train_data = f.get_processed_data('train.csv')
test_data = f.get_processed_data('test.csv')

# Split features and target variable
X = train_data.drop('RT', axis=1)  
y_pred= train_data['RT']

# Define the RidgeRegressor model
model = Ridge()

# Define the parameter grid for hyperparameter tuning
param_grid = {'alpha': np.logspace(-10, -3, 100)}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y_pred)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

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
plt.legend()
plt.show()