import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV 
from sklearn.linear_model import Lasso, Ridge
import matplotlib.pyplot as plt
import function as f

# Load data
# train_data = f.get_processed_data('train.csv')
# test_data = f.get_processed_data('test.csv')


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
train_data = f.clean_data(train_data_encoded, columns_to_drop)
test_data = f.clean_data(test_data_encoded, columns_to_drop)


def tune_model(model, data):

    # Create a KFold cross-validator with 20 folds
    kf = KFold(n_splits=20, shuffle=True, random_state=42)

    # Define the parameter grid for tuning alpha
    param_grid = {
        'alpha': np.logspace(-1, 5, num =30)
    }

    # Create a grid search with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='neg_root_mean_squared_error')

    # Fit the grid search to the data
    grid_search.fit(data.drop('RT', axis=1), data['RT'] )

    return grid_search

""" --------------------------------------------------- Lasso not working --------------------------------------------------- """

# Tune the LassoRegressor model 
# res1_lasso = tune_model(Lasso(max_iter = 10000), train_data)
# print("Best Alpha:", res1_lasso.best_estimator_.alpha, "Best Parameters:", res1_lasso.best_estimator_.coef_, "Best intercept:", res1_lasso.best_estimator_.intercept_)
# # Plot results
# plt.figure()
# plt.plot(np.logspace(-5, 7, num =30), -res1_lasso.cv_results_['mean_test_score'])
# plt.xscale('log')
# plt.xlabel('Alpha')
# plt.ylabel('RMSE')
# plt.title('Lasso Hyperparameter Tuning')
# plt.show()

""" --------------------------------------------------- Ridge Model --------------------------------------------------- """
X = train_data.drop('RT', axis=1)

res1_ridge = tune_model(Ridge(), train_data.iloc[:2800,:])
print("Best Alpha:", res1_ridge.best_estimator_.alpha, "Best Parameters:", res1_ridge.best_estimator_.coef_, "Best intercept:", res1_ridge.best_estimator_.intercept_)

pred = pd.DataFrame(res1_ridge.best_estimator_.predict(X.iloc[2800:,:]))

# f.make_submission(pred,"ridge_sub.csv")
f.test_error(train_data['RT'].iloc[2800:],pred)


""" --------------------------------------------------- Plots --------------------------------------------------- """


# Plot results
# plt.figure()
# plt.plot(np.logspace(-5, 7, num =30), -res1_ridge.cv_results_['mean_test_score'])
# plt.xscale('log')
# plt.xlabel('Alpha')
# plt.ylabel('RMSE')
# plt.title('Ridge Hyperparameter Tuning')
# plt.show()
