from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.model_selection import GridSearchCV 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import function as f 

# Load data
train_data, test_data = f.read_file('train.csv','test.csv')

# Drop categorical values
test_data = test_data.drop(['SMILES', 'mol','Compound'], axis=1)
train_data = train_data.drop(['SMILES','mol','Compound'], axis=1)

# Columns to encode
columns_to_encode = ['Lab']

# OneHotEncoder on train_data and test_data
train_data_encoded, test_data_encoded = f.apply_one_hot_encoding(train_data, test_data, columns_to_encode)

# Clean both train and test data using these columns
train_data, test_data = f.clean_data(train_data_encoded, test_data_encoded)

# Separate the features X and the target variable y
train_subset = int(len(train_data) * 0.8)
X,y = train_data.drop(['RT'], axis=1), train_data['RT']

# Separate train an test data
X_train, X_test, y_train, y_test = f.split_data(train_data)

""" # Apply PCA
variance_ratio = 0.989
X, test_data = f.apply_PCA(X, test_data, variance_ratio)

X_train = X.iloc[:train_subset, :]
X_test = X.iloc[train_subset:, :]
y_train = y.iloc[:train_subset]
y_test = y.iloc[train_subset:]

X_subset = pd.concat([X_train,y_train], axis=1).iloc[:train_subset,:]  """

# --------------------------------------------------- Ridge Model with tuning --------------------------------------------------- 

# Define the parameter grid for tuning alpha
param_grid = {
    'alpha': np.logspace(-1, 5, num =30)
}

# Model testing parameters 
submit = False
file_path = 'ridge_sub.csv'
plot = True

# Tune the Ridge Regressor model 
ridge_model = f.tune_model(Ridge(), train_data, param_grid, submit)

# Test the model
f.model_test(train_data,test_data,ridge_model.best_estimator_,submit,file_path,plot)

# --------------------------------------------------- Plots --------------------------------------------------- 

""" # Plot results
plt.figure()
plt.plot(np.logspace(-5, 7, num =30), -ridge_model.cv_results_['mean_test_score'])
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('RMSE')
plt.title('Ridge Hyperparameter Tuning')
plt.show() """

# --------------------------------------------------- Linear Regression not working --------------------------------------------------- 

"""
# Model testing parameters 
submit = False
file_path = 'linear_sub.csv'
plot = True


# Test the model
f.model_test(train_data,test_data,LinearRegression(),submit,file_path,plot)
"""

#--------------------------------------------------- Ridge Regression without tuning ---------------------------------------------------

"""
# Model testing parameters 
submit = False
file_path = 'ridge_sub.csv'
plot = True

# Test the model
f.model_test(train_data,test_data, Ridge(),submit,file_path,plot)

 """

# --------------------------------------------------- Lasso not working --------------------------------------------------- 

""" # Define the parameter grid for tuning alpha
param_grid = {
    'alpha': np.logspace(-1, 5, num =30)
}

# Tune the Lasso Regressor model 
lasso_model = f.tune_model(Lasso(max_iter = 10000), train_data, param_grid, submit)

# Test the model
f.model_test(train_data,test_data, lasso_model.best_estimator_ ,submit,file_path,plot)

# Plot results
plt.figure()
plt.plot(np.logspace(-5, 7, num =30), -lasso_model.cv_results_['mean_test_score'])
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('RMSE')
plt.title('Lasso Hyperparameter Tuning')
plt.show()  """


