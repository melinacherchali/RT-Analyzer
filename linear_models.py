from sklearn.linear_model import Ridge, LinearRegression, Lasso
import numpy as np
import pandas as pd
import function as f
import random
from sklearn.model_selection import train_test_split

# Set a random seed for reproducibility
random_seed = 2
np.random.seed(random_seed)
random.seed(random_seed)

# Model testing parameters 
submit = False
plot = True
select_features = False
PCA = True
CDDD = False

# Load data
train_data, test_data = f.read_file('train.csv','test.csv')

# Process and clean data
train_data, test_data = f.process_data(train_data,test_data, CDDD)

if PCA :
    # Apply PCA
    variance_ratio = 0.98
    X,y = train_data.drop(['RT'], axis=1), train_data['RT']
    X, test_data = f.apply_PCA(X, test_data, variance_ratio)

    train_data = pd.concat([X,y], axis=1)

# Split training_data in a training an test set 
X,y = train_data.drop(['RT'], axis=1), train_data['RT']
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# --------------------------------------------------- Ridge Model --------------------------------------------------- 
""" 
# Model testing parameters 
file_path = 'ridge_sub.csv'

# Define model
model_ridge = Ridge(alpha= 0.11, random_state=random_seed)

if not submit :
    f.train_and_evaluate_model(model_ridge, X_train, y_train, X_test, y_test, plot=False)
else :
    model_ridge.fit(X,y) 
    y_pred = pd.DataFrame(model_ridge.predict(test_data)) 
    f.make_submission(y_pred, file_path)
  """
# --------------------------------------------------- Lasso --------------------------------------------------- 
"""  
# Model testing parameters 
file_path = 'lasso.csv'

# Define the parameter grid for tuning alpha
model_lasso = Lasso(alpha= 0.001,max_iter=10000, random_state=random_seed)

if not submit :
    f.train_and_evaluate_model(model_lasso, X_train, y_train, X_test, y_test, plot=False)

    if select_features and not PCA :
        model_lasso.fit(X,y) 

        # Extract interesting features 
        feature_names = X.columns 
        selected_features = feature_names[model_lasso.coef_ != 0]

        # Convert to list
        selected_features_list = selected_features.tolist()

        # Print selected features
        print ("Selected Features != 0:", selected_features_list)
        print ("length", len(selected_features_list))

else :
    model_lasso.fit(X,y) 
    y_pred = pd.DataFrame(model_lasso.predict(test_data)) 
    f.make_submission(y_pred, file_path)
 """
# --------------------------------------------------- Linear Regression --------------------------------------------------- 

# Model testing parameters 
file_path = 'linear_sub.csv'

# Define the parameter grid for tuning alpha
model_lr = LinearRegression()

if not submit :
    f.train_and_evaluate_model(model_lr, X_train, y_train, X_test, y_test, plot=False)
else :
    model_lr.fit(X,y) 
    y_pred = pd.DataFrame(model_lr.predict(test_data)) 
    f.make_submission(y_pred, file_path)
 




