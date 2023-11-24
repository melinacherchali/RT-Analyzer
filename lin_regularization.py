import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV 
from sklearn.linear_model import Lasso, Ridge
import matplotlib.pyplot as plt
import function as f

# Load data
train_data = f.get_processed_data('train.csv')
test_data = f.get_processed_data('test.csv')

def tune_model(model, data):

    # Create a KFold cross-validator with 20 folds
    kf = KFold(n_splits=20, shuffle=True, random_state=42)

    # Define the parameter grid for tuning alpha
    param_grid = {
        'alpha': np.logspace(-1, 7, num =30)
    }

    # Create a grid search with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='neg_root_mean_squared_error')

    # Fit the grid search to the data
    grid_search.fit(data.sort_index(axis=1).drop('RT', axis=1), data['RT'] )

    return grid_search

#    ------  LASSO NOT WORKING :   ------ 

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

# Tune the Ridge Regressor model
X = train_data.sort_index(axis=1).drop('RT', axis=1)
res1_ridge = tune_model(Ridge(), train_data.iloc[:2800,:])
print("Best Alpha:", res1_ridge.best_estimator_.alpha, "Best Parameters:", res1_ridge.best_estimator_.coef_, "Best intercept:", res1_ridge.best_estimator_.intercept_)

pred = pd.DataFrame(res1_ridge.best_estimator_.predict(test_data))
print(pred)
f.make_submission(pred,"ridge_test.csv")

# Plot the predictions against actual values
# plt.figure()
# plt.scatter(pred,  train_data['RT'].iloc[2800:], label="Predictions vs Actual Values")
# plt.plot(np.arange(len(pred)), np.arange(len(pred)), c="black", ls="dashed", label="y=x")
# plt.xlabel("Predictions")
# plt.ylabel("Actual Values")
# plt.xlim(0, 5)
# plt.ylim(0, 5)
# plt.legend()
# plt.show() 



# Plot results
# plt.figure()
# plt.plot(np.logspace(-5, 7, num =30), -res1_ridge.cv_results_['mean_test_score'])
# plt.xscale('log')
# plt.xlabel('Alpha')
# plt.ylabel('RMSE')
# plt.title('Ridge Hyperparameter Tuning')
# plt.show()
