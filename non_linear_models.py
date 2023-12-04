from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.svm import NuSVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import pandas as pd
import numpy as np
import function as f 

# Load data
train_data, test_data = f.read_file('train.csv','test.csv')

# Process and clean data
train_data, test_data = f.process_data(train_data,test_data)

# Separate train and test data
train_subset = int(len(train_data) * 0.8)

X, y = train_data.drop('RT', axis=1), train_data['RT']
X_train, X_test, y_train, y_test = f.split_data(train_data)

# Apply PCA
variance_ratio = 0.989
X, test_data = f.apply_PCA(X, test_data, variance_ratio)

X_train = X.iloc[:train_subset, :]
X_test = X.iloc[train_subset:, :]
y_train = y.iloc[:train_subset]
y_test = y.iloc[train_subset:]

# --------------------------------------------------- Gradient boosting ---------------------------------------------------

# Model testing parameters
submit = False
file_path = 'GB_sub.csv'
plot = True

# Tune with Gradient boosting model
# model_GB = f.tune_model(XGBRegressor(), train_data, param_grid, submit)
# f.model_test(train_data,test_data,model_GB.best_estimator_,submit,file_path,plot)

param_dist = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],  # Include a wider range of values
    'n_estimators': [200, 500, 1000, 1500, 2000],  # Include higher values
    'max_depth': np.arange(2, 10, 1),  # Extend the range of max_depth
    'subsample': np.linspace(0.5, 1, 10),  # Increase the number of values
    'colsample_bytree': np.linspace(0.5, 1, 10),  # Increase the number of values
    'reg_alpha': [0, 0.01, 0.1, 0.5, 1, 2, 5],  # Include smaller values
    'reg_lambda': [0, 0.01, 0.1, 0.5, 1, 2, 5],  # Include smaller values
}
random_search = RandomizedSearchCV(XGBRegressor(), param_distributions=param_dist, n_iter=15, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)

if submit:
    X = train_data.iloc[:train_subset,:]
else :
    X = train_data

random_search.fit(X.drop('RT', axis=1), np.sqrt(X['RT']))
f.model_test(train_data,test_data,random_search.best_estimator_,submit,file_path,plot)
print("Best Parameters:", random_search.best_params_)


# --------------------------------------------------- Random Forest --------------------------------------------------- 

# Define the parameter grid
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_features': ['sqrt', 'log2', .5],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False]
# }

# Model testing parameters
submit = False
file_path = 'random_forest_sub.csv'
plot = True

# # Tune with Random Forest model
# rf_model = f.tune_model(RandomForestRegressor(), train_data, param_grid, submit)
# # rf_model = RandomForestRegressor(n_estimators=500)

# f.model_test(train_data,test_data,rf_model.best_estimator_,submit,file_path,plot)

param_dist = {
    'n_estimators': [500],
    'max_features': ['sqrt', 'log2', 1.0],
    'max_depth': [None, 10, 20, 30, 40],
    'bootstrap': [True]
}

# Use RandomizedSearchCV instead of GridSearchCV
# rf_model = RandomizedSearchCV(RandomForestRegressor(), param_distributions=param_dist, n_iter=20, cv=5, n_jobs=-1)

# Fit the model
X = train_data.iloc[:train_subset,:]
# rf_model.fit(X.drop('RT', axis=1), X['RT'])

# Get the best parameters
# f.model_test(train_data,test_data,rf_model.best_estimator_,submit,file_path,plot)

# --------------------------------------------------- SVM ---------------------------------------------------
""" 
# Usually best for classification problems 
model_SVM = NuSVR()

# Model testing parameters
submit = False
file_path = 'SVM_sub.csv'
plot = True
f.model_test(train_data,test_data,model_SVM,submit,file_path,plot) """