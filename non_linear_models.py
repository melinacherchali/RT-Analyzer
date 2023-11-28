from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV 
import pandas as pd
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
train_subset = int(len(train_data) * 0.8)

# Separate train and test data
X, y = train_data.drop('RT', axis=1), train_data['RT']
X_train, X_test, y_train, y_test = f.split_data(train_data)

""" # Apply PCA
variance_ratio = 0.989
X, test_data = f.apply_PCA(X, test_data, variance_ratio)

X_train = X.iloc[:train_subset, :]
X_test = X.iloc[train_subset:, :]
y_train = y.iloc[:train_subset]
y_test = y.iloc[train_subset:]
"""

# --------------------------------------------------- Random Forest --------------------------------------------------- 

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2', .5],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Model testing parameters
submit = False
file_path = 'random_forest_sub.csv'
plot = True

# Tune with Random Forest model
# rf_model = f.tune_model(RandomForestRegressor(), train_data, param_grid, submit)

rf_model = RandomForestRegressor(n_estimators=500)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
f.model_test(train_data,test_data,rf_model,submit,file_path,plot)


# Train the model to submit (on test set)
#rf_model.fit(X, y)

# Train the model 
# rf_model.fit(X_train,y_train)

# Evaluate the model to submit (on test_data)
#y_pred = pd.DataFrame(rf_model.predict(test_data))

# Evaluate the model (on X_test)
# y_pred = pd.DataFrame(rf_model.predict(X_test))

# Test error 
# f.test_error(y_test, y_pred, True)

# Make submission
#f.make_submission(y_pred, "random_forest_sub.csv")