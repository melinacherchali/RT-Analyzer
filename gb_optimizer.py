from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.svm import NuSVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import pandas as pd
import numpy as np
import function as f 
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error

# Load data
train_data, test_data = f.read_file('train.csv','test.csv')

# Process and clean data
train_data, test_data = f.process_data(train_data,test_data)

# Separate train and test data
train_subset = int(len(train_data) * 0.8)

X, y = train_data.drop('RT', axis=1), train_data['RT']
X_train, X_test, y_train, y_test = f.split_data(train_data)

# Apply PCA
# variance_ratio = 0.989
# X, test_data = f.apply_PCA(X, test_data, variance_ratio)

# X_train = X.iloc[:train_subset, :]
# X_test = X.iloc[train_subset:, :]
# y_train = y.iloc[:train_subset]
# y_test = y.iloc[train_subset:]

# --------------------------------------------------- Gradient boosting ---------------------------------------------------

# Model testing parameters
submit = False
file_path = 'GB_sub.csv'
plot = True

# Tune with Gradient boosting model
# Define the parameter grid for RandomizedSearchCV
# param_dist = {
#     'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
#     'n_estimators': [200, 500, 1000, 1500, 2000],
#     'max_depth': np.arange(2, 10, 1),
#     'subsample': np.linspace(0.5, 1, 10),
#     'colsample_bytree': np.linspace(0.5, 1, 10),
#     'reg_alpha': [0, 0.01, 0.1, 0.5, 1, 2, 5],
#     'reg_lambda': [0, 0.01, 0.1, 0.5, 1, 2, 5],
# }

param_dist = {
    'learning_rate': [0.1, 0.11, 0.12],
    'n_estimators': [500, 1000, 1500, 2000, 2500],
    'max_depth': np.arange(4, 7, 1),
    'subsample': np.linspace(0.84, 0.85, 10),
    'colsample_bytree':  np.linspace(0.8, 0.85, 10),
}

# Perform RandomizedSearchCV
search = RandomizedSearchCV(
    XGBRegressor(reg_alpha=1),
    param_distributions=param_dist,
    n_iter=15,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1
)

if submit :
    X = train_data
else : 
    X = train_data.iloc[:train_subset,:]


search.fit(X.drop('RT', axis=1), np.sqrt(X['RT']))


# Extract results and visualize
results = pd.DataFrame(search.cv_results_)

# Define the hyperparameters you want to visualize
hyperparameters = ['learning_rate', 'n_estimators', 'max_depth', 'subsample', 'colsample_bytree']

# Create separate subplots for each hyperparameter
fig, axes = plt.subplots(nrows=len(hyperparameters), figsize=(10, 4 * len(hyperparameters)))

for i, param in enumerate(hyperparameters):
    sns.lineplot(x=f'param_{param}', y=np.sqrt(-results['mean_test_score']), data=results, ax=axes[i])
    axes[i].set_title(f'{param} vs RMSE')
    axes[i].set_xlabel(param)
    axes[i].set_ylabel('Root Mean Squared Error (RMSE)')

plt.tight_layout()
plt.show()

# Print the best hyperparameters and corresponding RMSE
best_params = search.best_params_
best_rmse = np.sqrt(-search.best_score_)
print("Best Hyperparameters:", best_params)
print("Best RMSE:", best_rmse)

f.model_test(train_data,test_data,search,submit,file_path,plot)
