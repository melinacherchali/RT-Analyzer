from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.svm import NuSVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import pandas as pd
import numpy as np
import function as f 
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


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
train_subset = int(len(train_data) * 0.8)

# Separate train and test data
X, y = train_data.drop('RT', axis=1), train_data['RT']
X_train, X_test, y_train, y_test = f.split_data(train_data)


# --------------------------------------------------- Gradient boosting ---------------------------------------------------

# Model testing parameters
submit = False
file_path = 'GB_sub.csv'
plot = True

# Tune with Gradient boosting model
# model_GB = f.tune_model(XGBRegressor(), train_data, param_grid, submit)
# f.model_test(train_data,test_data,model_GB.best_estimator_,submit,file_path,plot)



# Create a pipeline with PCA and XGBoost
pipeline = Pipeline([
    ('pca', PCA()),  # PCA will be part of the pipeline
    ('xgb', XGBRegressor())
])

param_dist = {
    'pca__n_components': [0.98, 0.99, 0.995, 0.997],  # You can adjust these ratios
    'xgb__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'xgb__n_estimators': [200, 500, 1000, 1500, 2000],
    'xgb__max_depth': np.arange(2, 10, 1),
    'xgb__subsample': np.linspace(0.5, 1, 10),
    'xgb__colsample_bytree': np.linspace(0.5, 1, 10),
    'xgb__reg_alpha': [0, 0.01, 0.1, 0.5, 1, 2, 5],
    'xgb__reg_lambda': [0, 0.01, 0.1, 0.5, 1, 2, 5]
}

if submit:
    X = train_data.iloc[:train_subset,:]
else :
    X = train_data
    
# Use RandomizedSearchCV
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=15, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)

# Fit the model with early stopping
eval_set = [(X.drop('RT', axis=1), X['RT'])]  # Use your validation set here
random_search.fit(X.drop('RT', axis=1), y_train, xgb__eval_metric="rmse", xgb__eval_set=eval_set, xgb__early_stopping_rounds=10, xgb__verbose=True)

# Fit the model
# random_search.fit(X.drop('RT', axis=1), X['RT'])

# Print the best parameters
print("Best Parameters:", random_search.best_params_)

# Retrieve the best PCA ratio
best_pca_ratio = random_search.best_params_['pca__n_components']

# Apply PCA with the best ratio to the training and test data
    
X, test_data = f.apply_PCA(X, test_data, best_pca_ratio)
X_train = X.iloc[:train_subset, :]
X_test = X.iloc[train_subset:, :]

f.model_test(train_data,test_data,random_search.best_estimator_,submit,file_path,plot)
print("Best Parameters:", random_search.best_params_)