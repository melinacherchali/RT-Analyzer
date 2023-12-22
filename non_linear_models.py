from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import NuSVR
from sklearn.model_selection import train_test_split
import function as f 
import pandas as pd
import numpy as np
import random

# Set a random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

# Model testing parameters 
submit = False
plot = True
CDDD = False

# Load data
train_data, test_data = f.read_file('train.csv','test.csv')

# Process and clean data
train_data, test_data = f.process_data(train_data,test_data, CDDD)

# Split training_data into training and test sets
X, y = train_data.drop('RT', axis=1), train_data['RT']
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, shuffle=True)

# --------------------------------------------------- Gradient boosting ---------------------------------------------------
  
# Model testing parameters
file_path = 'GB_sub.csv'

# Define model
model_GB =  XGBRegressor(subsample= 0.843, 
                  reg_lambda= 1, 
                  reg_alpha= 0, 
                  n_estimators= 2500, 
                  max_depth= 4, 
                  learning_rate= 0.11, 
                  colsample_bytree= 0.8055,
                  seed=random_seed)

if not submit :
    f.train_and_evaluate_model(model_GB, X_train, y_train, X_test, y_test, plot=False)
else :
    model_GB.fit(X,y) 
    y_pred = pd.DataFrame(model_GB.predict(test_data)) 
    f.make_submission(y_pred, file_path)
  

# --------------------------------------------------- Random Forest --------------------------------------------------- 
"""  
# Model testing parameters
file_path = 'random_forest_sub.csv'

# Define model
model_rf = RandomForestRegressor(n_estimators=200, max_depth= 300, random_state=random_seed)

if not submit :
    f.train_and_evaluate_model(model_rf, X_train, y_train, X_test, y_test, plot=False)
else :
    model_rf.fit(X,y) 
    y_pred = pd.DataFrame(model_rf.predict(test_data)) 
    f.make_submission(y_pred, file_path)
   """
# --------------------------------------------------- SVM ---------------------------------------------------
"""    
# Model testing parameters
file_path = 'SVM_sub.csv'

# Define model
model_SVM = NuSVR()

if not submit :
    f.train_and_evaluate_model(model_SVM, X_train, y_train, X_test, y_test, plot=False)
else :
    model_SVM.fit(X,y) 
    y_pred = pd.DataFrame(model_SVM.predict(test_data)) 
    f.make_submission(y_pred, file_path)
 
 """